from typing import Optional
import anndata
import networkx as nx
import numpy as np
import pandas as pd

def validate_against_base_network(adata: anndata.AnnData,
                                    condition_label: str,
                                    causal_network_label: str = 'causal_networks',
                                    base_network_label: str = 'base_networks', 
                                    feasible_pairs: Optional[list] = None,
                                    celltype_sep_new: str = '-',
                                    celltype_sep_old: str = ' ',
                                    node_sep: str = '_'):
    """
    Validate the learned causal signaling network from UT-IGSP by checking interactions
    against the original base network. This removes any inferred edges that cannot
    occur via cell-cell communication.

    Parameters
    ----------
    adata
        The annotated dataframe (typically from Scanpy) of the single-cell data.
        Must contain inferred causal signaling networks between cell-type-ligand
        pairs.

    condition_label 
        The label in adata.obs which we use to partition the data.

    causal_network_label
        The label for which inferred causal signaling output is stored in adata.uns

    base_network_label
        The label for which the base network and original CCC inference are stored in adata.uns

    celltype_sep_new
        String used to replace celltype_sep_old (spaces/underlines) in original cell type
        annotation of scRNA-seq, to make formatting easier for UT-IGSP
    
    celltype_sep_old
        String that was originally replaced in cell type annotation, like spaces/underlines.
        We need to  convert cell type labels back to match them to the original CCC annotation

    node_sep
        String separator used to join cell type and ligand annotations when inferring the causal
        signaling network.

    Returns
    -------
    causal_output_df
        Pandas dataframe of causal interactions and their relevant ligand-receptor interactions,
        their condition, and the frequency of interaciton as, inferred from bootstrap aggregation.
        Stored in adata.uns[causal_network_label]['interactions']. This dataframe can be used for
        further validation against a tool like NicheNet.

    """

    # Get all possible conditions
    conditions = adata.obs[condition_label].unique().tolist()

    # Get the original CCC output and base networks
    ccc_output = adata.uns[base_network_label]['ccc_output']
    base_networks = adata.uns[base_network_label]['networks']

    # Get the nodes and adjacency of the inferred causal network
    celltype_ligands = list(adata.uns[causal_network_label]['celltype_ligands'])
    causal_adjacency = adata.uns[causal_network_label]['causal_adjacency']

    sources = []
    targets = []
    ligands_sources = []
    ligands_targets = []
    edge_frequencies = []
    receptors_targets = []
    conditions_found = []

    nonzero_rows, nonzero_cols = causal_adjacency.nonzero()

    for j in range(len(nonzero_rows)):

        node_1 = celltype_ligands[nonzero_rows[j]]
        node_2 = celltype_ligands[nonzero_cols[j]]

        node_1_celltype = node_1.split(node_sep)[0].replace(celltype_sep_new, celltype_sep_old) # Convert back to the original labels
        node_1_ligand = node_1.split(node_sep)[1]

        node_2_celltype = node_2.split(node_sep)[0].replace(celltype_sep_new, celltype_sep_old) # Convert back to the original labels
        node_2_ligand = node_2.split(node_sep)[1]

        if feasible_pairs:

            if ( ( (node_1_celltype, node_2_celltype) in feasible_pairs)\
                 |((node_2_celltype, node_1_celltype) in feasible_pairs) ) :

                relevant_conditions = []

                for condition in conditions:

                    if ((node_1, node_2) in base_networks[condition]['edges']):
                        relevant_conditions.append(condition)

                    # Get the relevant interactions and thus receptors
                    relevant_interactions = []

                    if len(relevant_conditions) == len(conditions):

                        relevant_conditions = ['Both']

                        if len(conditions) > 2: # Change just to be pedantic
                            relevant_conditions = ['All']

                    if ('All' in relevant_conditions)|('Both' in relevant_conditions):
                        relevant_interactions = pd.concat([ccc_output[cond][(ccc_output[cond]['source'] == node_1_celltype)
                                                                                    &(ccc_output[cond]['target'] == node_2_celltype)
                                                                                    &(ccc_output[cond]['ligand'] == node_1_ligand)] for cond in conditions])
                    else:
                        relevant_interactions = pd.concat([ccc_output[cond][(ccc_output[cond]['source'] == node_1_celltype)
                                                                                    &(ccc_output[cond]['target'] == node_2_celltype)
                                                                                    &(ccc_output[cond]['ligand'] == node_1_ligand)] for cond in relevant_conditions])

                    receptors = []

                    unique_interactions = relevant_interactions['interaction_name_2'].unique()

                    for interaction in unique_interactions:
                        receptor_split = interaction.split(' - ')[1].strip('()').split('+')
                        for sub in receptor_split:
                            if sub not in receptors:
                                receptors.append(sub)

                    # Join the receptors
                    receptors_joined = '_'.join(receptors)
                    
                    if receptors_joined: # Only add the row if the receptors have been identified
                        sources.append(node_1_celltype)
                        targets.append(node_2_celltype)
                        ligands_sources.append(node_1_ligand)
                        ligands_targets.append(node_2_ligand)
                        edge_frequencies.append(causal_adjacency[nonzero_rows[j], nonzero_cols[j]])
                        conditions_found.append('_'.join(relevant_conditions))
                        receptors_targets.append(receptors_joined)

        else:

            relevant_conditions = []

            for condition in conditions:

                if ((node_1, node_2) in base_networks[condition]['edges']):
                    
                    relevant_conditions.append(condition)

                if len(relevant_conditions) == len(conditions):

                    relevant_conditions = ['Both']
                    
                    if len(conditions) > 2: # Change just to be pedantic
                        relevant_conditions = ['All']

                # Get the relevant interactions and thus receptors
                relevant_ccc = []

                if ('All' in relevant_conditions)|('Both' in relevant_conditions):
                        relevant_ccc = [ccc_output[cond][(ccc_output[cond]['source'] == node_1_celltype)
                                        &(ccc_output[cond]['target'] == node_2_celltype)
                                        &(ccc_output[cond]['ligand'] == node_1_ligand)] for cond in conditions]

                else:
                    relevant_ccc = [ccc_output[cond][(ccc_output[cond]['source'] == node_1_celltype)
                                    &(ccc_output[cond]['target'] == node_2_celltype)
                                    &(ccc_output[cond]['ligand'] == node_1_ligand)] for cond in relevant_conditions]

                if relevant_ccc:                                                            
                    relevant_interactions = pd.concat(relevant_ccc)

                    receptors = []

                    unique_interactions = relevant_interactions['interaction_name_2'].unique()

                    for interaction in unique_interactions:
                        receptor_split = interaction.split(' - ')[1].strip('()').split('+')
                        for sub in receptor_split:
                            if sub not in receptors:
                                receptors.append(sub)

                    # Join the receptors
                    receptors_joined = '_'.join(receptors)
                    
                    if receptors_joined: # Only add the row if the receptors have been identified
                        sources.append(node_1_celltype)
                        targets.append(node_2_celltype)
                        ligands_sources.append(node_1_ligand)
                        ligands_targets.append(node_2_ligand)
                        edge_frequencies.append(causal_adjacency[nonzero_rows[j], nonzero_cols[j]])
                        conditions_found.append('_'.join(relevant_conditions))
                        receptors_targets.append(receptors_joined)

    # Save the CSV
    causal_output_df = pd.DataFrame(data={'Source':sources, 'Ligand1':ligands_sources,
                                    'Target':targets, 'Ligand2':ligands_targets, 'Receptor':receptors_targets,
                                    condition_label: conditions_found, 'Frequency':edge_frequencies})

    adata.uns[causal_network_label]['interactions'] = causal_output_df