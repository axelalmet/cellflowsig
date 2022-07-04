import numpy as np
import pandas as pd
import anndata as ad

def construct_base_networks(adata, ccc_outputs, condition_label,
                            method='cellchat', feasible_pairs=[],
                            celltype_sep_old=' ', celltype_sep_new='-', node_sep='_',
                            base_network_label='base_networks',
                            pval_cutoff=0.05):

    ccc_methods = ['cellchat', 'cellphonedb', 'squidpy']

    if method not in ccc_methods:

        print('Method needs to be one of cellchat, cellphonedb, or squidpy.')

    else:

        if method == 'cellchat':
            
            adata.uns[base_network_label] = construct_base_networks_from_cellchat(adata, ccc_outputs, feasible_pairs, condition_label, celltype_sep_old, celltype_sep_new, node_sep)

        elif method == 'cellphonedb':

            adata.uns[base_network_label] = construct_base_networks_from_cellphonedb(adata, ccc_outputs, feasible_pairs, condition_label, celltype_sep_old, celltype_sep_new, node_sep, pval_cutoff)

        else: # Should be squidpy

            adata.uns[base_network_label] = construct_base_networks_from_squidpy(adata, ccc_outputs, feasible_pairs, condition_label, celltype_sep_old, celltype_sep_new, node_sep, pval_cutoff)

def construct_base_networks_from_cellchat(adata, cellchat_outputs, condition_label, feasible_pairs, celltype_sep_old, celltype_sep_new, node_sep):

    conditions = adata.obs[condition_label].unique().tolist()

    # Store the possible causal edges for each condition, as well as the union of all cell-type-ligand pairs considered
    base_networks = {condition:{'nodes':[], 'edges':[]} for condition in conditions}
    considered_celltype_ligands = []

    # Construct the possible causal edges between cell-type-ligand pairs
    for i in range(len(conditions)):

        condition = conditions[i]
        ccc_output = cellchat_outputs[condition]

        possible_edges = []

        for index, row in ccc_output.iterrows():

            celltype_A = row['source']
            celltype_B = row['target']
            ligand_A = row['ligand']

            
            # Correct for the ligand and receptor names if they're not found in the data
            if ligand_A not in adata.var_names:
                ligand_A = row['interaction_name_2'].split(' - ')[0]

            # Define the celltype ligand pairs
            celltype_ligand_A = celltype_A.replace(celltype_sep_old, celltype_sep_new) + node_sep + ligand_A

            # Get all of the secondary interactions with cell type B as the sender cell type
            celltype_B_targets = ccc_output[ccc_output['source'] == celltype_B]

            for target_index, target_row in celltype_B_targets.iterrows():

                ligand_B = target_row['ligand']
                
                # Correct for the ligand and receptor names if they're not found in the data
                if ligand_B not in adata.var_names:
                    ligand_B = target_row['interaction_name_2'].split(' - ')[0]

                celltype_ligand_B = celltype_B.replace(celltype_sep_old, celltype_sep_new) + node_sep + ligand_B 

                if index != target_index: # So that we don't store the same interaction

                    # Add the celltype-ligand pair to the condition-specific list of edges
                    if feasible_pairs: # If a list of feasible pairs has been specified

                        if ((celltype_ligand_A, celltype_ligand_B) in feasible_pairs)|\
                         ((celltype_ligand_B, celltype_ligand_A) in feasible_pairs):

                            if (celltype_ligand_A, celltype_ligand_B) not in possible_edges:
                                possible_edges.append((celltype_ligand_A, celltype_ligand_B))

                            # Also add the pairs to the union list of possible nodes
                            if celltype_ligand_A not in considered_celltype_ligands:
                                considered_celltype_ligands.append(celltype_ligand_A)

                            if celltype_ligand_B not in considered_celltype_ligands:
                                considered_celltype_ligands.append(celltype_ligand_B)

                    else: # Otherwise we can just add the edge and update the joint nodes list

                        if (celltype_ligand_A, celltype_ligand_B) not in possible_edges:
                            possible_edges.append((celltype_ligand_A, celltype_ligand_B))

                        # Also add the pairs to the union list of possible nodes
                        if celltype_ligand_A not in considered_celltype_ligands:
                            considered_celltype_ligands.append(celltype_ligand_A)

                        if celltype_ligand_B not in considered_celltype_ligands:
                            considered_celltype_ligands.append(celltype_ligand_B)
                    

        base_networks[condition]['edges'] = possible_edges

        considered_celltype_ligands = sorted(considered_celltype_ligands) # Sort the nodes

        # Add the list of nodes to each condition-specific candidate network
        for condition in conditions:

            base_networks[condition]['nodes'] = considered_celltype_ligands

        # Finally construct the "union" candidate network across all conditions
        base_networks['joined']['nodes'] = considered_celltype_ligands
        base_networks['joined']['edges'] = list(set.union(*map(set, [base_networks[condition]['edges'] for condition in conditions])))

    # Define as dictionary
    base_networks = {'ccc_output':cellchat_outputs, 'celltype_ligands':considered_celltype_ligands, 'base_networks':base_networks}
    return base_networks

def construct_base_networks_from_cellphonedb(adata, cellphonedb_outputs, feasible_pairs, condition_label, celltype_sep_old, celltype_sep_new, node_sep, pval_cutoff):

    converted_cellphonedb_output = {}

    for condition in cellphonedb_outputs:

        output = cellphonedb_outputs[condition]

        # Get the means and p-values for each cellphonedb output
        output_means = output['means']
        output_pvalues = output['pvalues']

        # Get the interacting pairs
        interacting_pairs = output_means.columns[11:]

        interaction_names = []
        ligands = []
        receptors = []
        sources = []
        targets = []
        probs = []
        pvals = []
        evidence = []

        for index, row in output_pvalues.iterrows():

            for pair in interacting_pairs:

                # Get the p-value
                if row[pair] < pval_cutoff:

                    # Split the ligand-receptor pair
                    interaction_name = row['interacting_pair']
                    ligand, receptor = interaction_name.split('_')

                    # Split the source-target pair
                    source, target = pair.split('|')

                    # Get the interaction score 
                    interaction_score = output_means.loc[index, pair]

                    # Get the p-value
                    pval = row[pair]

                    ref = row['annotation_strategy']

                    interaction_names.append(interaction_name)
                    ligands.append(ligand)
                    receptors.append(receptor)
                    sources.append(source)
                    targets.append(target)
                    probs.append(interaction_score)
                    pvals.append(pval)
                    evidence.append(ref)
                
            # Construct dataframes similar to CellChat output
            converted_output = pd.DataFrame(data={'source':sources, 'target':targets,\
                                        'ligand':ligands, 'receptor':receptors,\
                                        'prob':probs, 'pval':pvals,\
                                        'interaction_name':interaction_names,\
                                        'evidence':evidence})

            converted_cellphonedb_output[condition] = converted_output
                                
        # Run the CellChat method
        base_networks = construct_base_networks_from_cellchat(adata, converted_cellphonedb_output, feasible_pairs, condition_label, celltype_sep_old, celltype_sep_new, node_sep)
        return base_networks

def construct_base_networks_from_squidpy(adata, squidpy_outputs, feasible_pairs, condition_label, celltype_sep_old, celltype_sep_new, node_sep, pval_cutoff):

    converted_squidpy_output = {}

    for condition in squidpy_outputs:

        output = squidpy_outputs[condition]

        interaction_names = []
        ligands = []
        receptors = []
        sources = []
        targets = []
        probs = []
        pvals = []
        evidence = []

        for index, row in output['pvalues'].iterrows():

            non_nan_indices = np.where(~np.isnan(row.to_numpy()))[0] # Only consider where there's non-nan interactions

            if len(non_nan_indices):

                # Get the relevant data
                ligand, receptor = index # Ligand and receptors
                interaction_name = '_'.join(index) # Joined interactions
                sources_targets = output['means'].columns[non_nan_indices]
                interaction_sources = [pair[0] for pair in sources_targets]
                interaction_targets = [pair[1] for pair in sources_targets]
                means = output['means'].loc[index[0], index[1]][non_nan_indices] # Scores
                pvalues = output['pvalues'].loc[index[0], index[1]][non_nan_indices] # P-values
                ref = output['metadata']['sources'].loc[index[0], index[1]]
                
                # Update the lists
                for i in range(len(non_nan_indices)):

                    if pvalues[i] < pval_cutoff:

                        interaction_names.append(interaction_name)
                        ligands.append(ligand)
                        receptors.append(receptor)
                        sources.append(interaction_sources[i])
                        targets.append(interaction_targets[i])
                        probs.append(means[i])
                        pvals.append(pvalues[i])
                        evidence.append(ref)
                
            # Construct dataframes similar to CellChat output
            converted_output = pd.DataFrame(data={'source':sources, 'target':targets,\
                                        'ligand':ligands, 'receptor':receptors,\
                                    'prob':probs, 'pval':pvals,\
                                    'interaction_name':interaction_names,\
                                    'evidence':evidence})

            converted_squidpy_output[condition] = converted_output
                                
        # Run the CellChat method
        base_networks = construct_base_networks_from_cellchat(adata, converted_squidpy_output, feasible_pairs, condition_label, celltype_sep_old, celltype_sep_new, node_sep)
        return base_networks
