import scanpy as sc
import networkx as nx
from scipy.sparse import issparse
import numpy as np
import random as rm
from causaldag import unknown_target_igsp
from causaldag import partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester
from causaldag import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester
from graphical_models import DAG
from sklearn.utils import safe_mask
import multiprocessing as mp
from timeit import default_timer as timer
from functools import reduce

def convert_adjacency_to_network(adata):

def learn_causal_network(adata, condition_label, control_label, causal_network_label, celltype_ligands_label, base_network_label, n_cores, n_bootstraps, n_shuffles, alpha_ci, alpha_inv):
    
    # Extract the control and perturbed samples
    conditions = adata.obs[condition_label].unique().tolist()
    control_sample = adata[adata.obs[condition_label] == control_label].obsm[celltype_ligands_label] # Define the control data
    perturbed_samples = [adata[adata.obs[condition_label] == cond].obsm[celltype_ligands_label]  for cond in conditions if cond != control_label] # Get the perturbed data

    celltype_ligands = adata.uns[base_network_label]['celltype_ligands']

    # Define the joined candidate network
    base_network_edges = adata.uns[base_network_label]['joined']['edges']

    # Define the sampling step function where we input the initial list of permutations
    def run_utigsp(control_sample, perturbed_samples, celltype_ligands, base_edges, n_shuffles, alpha, alpha_inv, seed):

        # Reseed the random number generator
        np.random.seed(seed) # Set the seed for reproducibility reasons

        # Get the number of samples for each dataframe  
        num_samples_control = control_sample.shape[0]
        num_samples_perturbed = [sample.shape[0] for sample in perturbed_samples]

        # Subsample with replacement
        subsamples_control = np.random.choice(num_samples_control, num_samples_control)
        subsamples_perturbed = [np.random.choice(num_samples, num_samples) for num_samples in num_samples_perturbed]

        control_resampled = control_sample[safe_mask(control_sample, subsamples_control), :]
        perturbed_resampled = []

        for i in range(len(perturbed_samples)):
            num_subsamples = subsamples_perturbed[i]
            perturbed_sample = perturbed_samples[i]

            resampled = perturbed_sample[safe_mask(num_subsamples, num_subsamples), :]
            perturbed_resampled.append(resampled)

        # We need to subset the gene expression matrices for ligands with non-zero standard deviation in BOTH cases
        control_resampled_std = control_resampled.std(0)
        perturbed_resampled_std = [sample.std(0) for sample in perturbed_resampled]

        nonzero_ligand_indices_control = control_resampled_std.nonzero()[0]
        nonzero_ligand_indices_perturbed = [resampled_std.nonzero()[0] for resampled_std in perturbed_resampled_std]

        nonzero_ligand_indices = reduce(np.intersect1d, (nonzero_ligand_indices_control, *nonzero_ligand_indices_perturbed))

        # Subset based on the ligands with zero std in both cases
        considered_celltype_ligands = list(celltype_ligands[nonzero_ligand_indices])
        nodes = set(considered_celltype_ligands)

        control_resampled = control_resampled[:, nonzero_ligand_indices]

        for i in range(len(perturbed_resampled)):
            resampled = perturbed_resampled[i]
            perturbed_resampled[i] = resampled[:, nonzero_ligand_indices]

        # We now need to set a new list of permutations
        initial_random_dags = []

        for iter in range(n_shuffles):
            
            # Create the initial DAG
            init_dag = nx.DiGraph()
            init_dag.add_nodes_from(considered_celltype_ligands)
            
            # Shuffle the edges
            rm.shuffle(base_edges)
            
            # Add the edges        
            for edge in base_edges:
                if (edge[0] in considered_celltype_ligands)&(edge[1] in considered_celltype_ligands): # Only take the accepted ligands
                    init_dag.add_edge(*edge)
                    # Check if it's a DAG
                    if not nx.is_directed_acyclic_graph(init_dag):
                        init_dag.remove_edge(*edge)
                    
            # Add the DAG to the list
            initial_random_dags.append(init_dag)

        init_perms = [DAG.topological_sort(DAG.from_nx(dag)) for dag in initial_random_dags]

        # Convert the permutations in terms of node indices
        converted_perms = []

        for perm in init_perms:
            converted_perm = [considered_celltype_ligands.index(node) for node in perm]
            if converted_perm not in converted_perms:
                converted_perms.append(converted_perm)

        ### Run UT-IGSP using partial correlation  

        # Form sufficient statistics using partial correlation (assumes linear Gaussian model)
        obs_suffstat = partial_correlation_suffstat(control_resampled, invert=True)
        invariance_suffstat = gauss_invariance_suffstat(control_resampled, perturbed_resampled)

        # Create conditional independence tester and invariance tester
        ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha)
        invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)

        # Assume unknown interventions for UT-IGSP
        setting_list = [dict(known_interventions=[]) for _ in perturbed_resampled]

        ## Run UT-IGSP by considering all possible initial permutations
        est_dag_allperms, est_targets_list_allperms = unknown_target_igsp(setting_list,
                                                                            nodes,
                                                                            ci_tester,
                                                                            invariance_tester,
                                                                            nruns=len(converted_perms),
                                                                            initial_permutations=converted_perms)

        adjacency_dag_allperms = est_dag_allperms.to_amat()[0]

        perturbed_targets_list = []
        
        for i in range(len(est_targets_list_allperms)):
            targets_list = list(est_targets_list_allperms[i])
            targets_ligand_indices = [celltype_ligands.index(considered_celltype_ligands[target]) for target in targets_list]
            perturbed_targets_list.append(targets_ligand_indices)

        return {'nonzero_ligand_indices':nonzero_ligand_indices,
                'adjacency_dag_allperms':adjacency_dag_allperms,
                'perturbed_targets_indices':perturbed_targets_list}

    def main():

        # Randomly shuffle to edges to generate initial permutations for initial DAGs
        bagged_adjacency_dag = np.zeros((len(celltype_ligands), len(celltype_ligands)))

        bagged_intervention_targets = [np.zeros(len(celltype_ligands)) for sample in perturbed_samples]

        # Define the multiprocessing 
        pool = mp.Pool(n_cores)

        start = timer()

        print(f'starting computations on {n_cores} cores')


        args_allperms = [(control_sample, perturbed_samples, 
                            celltype_ligands,
                            base_network_edges, n_shuffles,
                            alpha_ci, alpha_inv,
                            boot) for boot in range(n_bootstraps)]

        bootstrap_results = pool.starmap(run_utigsp, args_allperms)

        end = timer()

        print(f'elapsed time: {end - start}')

        # Sum the results for UT-IGSP with initial permutations
        for res in bootstrap_results:

            nz_indices = res['nonzero_celltypeligand_indices']
            adjacency = res['adjacency_dag']
            int_indices = res['perturbed_targets_indices']

            # Update the bagged adjacency
            bagged_adjacency_dag[np.ix_(nz_indices, nz_indices)] += adjacency

            # Update the intervention targets
            for i in range(len(int_indices)):

                nonzero_int_indices = int_indices[i]
                intervention_targets = bagged_intervention_targets[i]
                intervention_targets[nonzero_int_indices] += 1
                bagged_intervention_targets[i] = intervention_targets

        # Average the adjacencies
        bagged_adjacency_dag /= float(n_bootstraps)

        # Average the intervened targets
        for i in range(len(int_indices)):

            intervention_targets = bagged_intervention_targets[i]
            intervention_targets /= float(n_bootstraps) # Average the results
            bagged_intervention_targets[i] = intervention_targets

        return {'adjacency': bagged_adjacency_dag, 'perturbed_targets':bagged_intervention_targets}

    # Run the code
    if __name__ == '__main__': 
        learned_network_results = main()

        # Store the learned network, which is an averaged adjacency matrix and a list of peturbed indices
        adata.uns[causal_network_label] = learned_network_results