import scanpy as sc
import networkx as nx
import numpy as np
import pandas as pd

def validate_against_base_network(adata, causal_network_label, base_network_label, celltype_label, condition_label, celltype_sep_new = '-', celltype_sep_old = ' ', node_sep='_'):

    # Get all possible conditions
    conditions = adata.obs[condition_label].unique().tolist()

    # Get the original CCC output and base networks
    ccc_output = adata.uns[base_network_label]['ccc_output']
    base_networks = adata.uns[base_network_label]['base_networks']

    # Get the nodes and adjacency of the inferred causal network
    celltype_ligands = list(adata.uns[causal_network_label]['celltype_ligands'])
    causal_adjacency = adata.uns[causal_network_label]['adjacency']

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

        node_1_celltype = node_1.split(node_sep)[0]
        node_1_ligand = node_1.split(node_sep)[1]

        node_2_celltype = node_2.split(node_sep)[0]
        node_2_ligand = node_2.split(node_sep)[1]

        relevant_conditions = []
        for condition in conditions:
            if ((node_1, node_2) in base_networks[condition]['edges']):
                relevant_conditions.append(condition)

            # Get the relevant interactions and thus receptors
            relevant_interactions = []

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

    # Save the CSV
    causal_output_df = pd.DataFrame(data={'Source':sources, 'Ligand1':ligands_sources,
                                    'Target':targets, 'Ligand2':ligands_targets, 'Receptor':receptors_targets,
                                    condition_label: conditions_found, 'Frequency':edge_frequencies})

    adata.uns[causal_network_label]['interactions'] = causal_output_df