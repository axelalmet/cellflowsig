from typing import Optional, List
from matplotlib import pyplot as plt
import anndata
import networkx as nx
import numpy as np
import random as rm
import itertools as it
import pandas as pd
from scipy.spatial import distance_matrix
import scanpy as sc
import os

def construct_directed_network(adata: anndata.AnnData, 
                            network_label: str,
                            node_sep: str = '_'):

    # Initialise the network
    directed_network = nx.DiGraph()

    # Get the cell type ligands
    celltype_ligands = adata.uns[network_label]['celltype_ligands']
    celltypes = sorted(list(set([pair.split(node_sep)[0] for pair in celltype_ligands])))
    ligands = sorted(list(set([pair.split(node_sep)[1] for pair in celltype_ligands])))

    # Add the nodes, taking care to order layers by ligands
    for celltype, ligand in it.product(celltypes, ligands):

        celltype_ligand = celltype + node_sep + ligand
        layer_index = celltypes.index(ligand)
        directed_network.add_node(celltype_ligand, layer=layer_index)

    # Add the edges now
    edge_sets = adata.uns[network_label]['networks']

    all_edges = list(set.union(*map(set, [edge_sets[condition]['edges'] for condition in edge_sets])))

    for edge in all_edges:

        source_pair, target_pair = edge

        # Determine the cell type source and target
        source = source_pair.split(node_sep)[0]
        target = target_pair.split(node_sep)[0]

        directed_network.add_edge(edge)
        
        # Annotate the edge with source and target
        directed_network.edges[source_pair, target_pair]['source'] = source
        directed_network.edges[source_pair, target_pair]['target'] = target

        # Get the conditions to which the edge belongs
        relevant_conditions = []
        for condition in edge_sets:

            if edge in edge_sets[condition]['edges']:

                relevant_conditions.append(condition)

        conditions_found = '_'.join(sorted(relevant_conditions))

        # Add the condition to the network now
        directed_network.edges[source_pair, target_pair]['condition'] = conditions_found

        # Now add the weights
        # If we're looking at the causal network, the weight is the bootstrapped edge frequency
        if 'causal_adjacency' in adata.uns[network_label]:

            causal_adjacency = adata.uns[network_label]['causal_adjacency']
            source_index = celltype_ligands.index(source_pair)
            target_index = celltype_ligands.index(target_pair)

            directed_network.edges[source_pair, target_pair]['weight'] = causal_adjacency[source_index, target_index]

        else: # We assume uniform weights of the base network

            directed_network.edges[source_pair, target_pair]['weight'] = 1.0

        

    return directed_network

def plot_clustered_network(adata: anndata.AnnData, 
                            network_label: str = 'causal_networks',
                            label_nodes: bool = True, 
                            output_name: Optional[str] = None,
                            celltype_sep_old: str =' ',
                            celltype_sep_new: str = '-',
                            node_sep: str = '_',
                            edge_type: str = 'condition',
                            circle_radius_outer: float = 7.0,
                            circle_radius_inner: float = 2.0,
                            ligand_label_shift: float = 1.1,
                            celltype_label_shift: float = 1.05,
                            node_size: int = 50,
                            node_border_scale: float = 2.0,
                            node_color_map: str = 'tab10',
                            edge_color_map: str = 'Set3',
                            edge_width_scale: float = 1.0,
                            arrow_size: int = 15,
                            edge_alpha: float = 0.6,
                            font_size: int = 12, 
                            xmargin: float = 0.2,
                            ymargin: float = 0.1):

    if network_label not in ['base_networks', 'causal_networks']:

        print("Current network options only include 'base_networks' or 'causal_networks'.")

    else:

        if edge_type not in ['condition', 'source', 'target']:

            print('Edge type not supported. Must be one of: source, target, condition')

        else:

            # Construct the directed graph where nodes are marked by 'layers' corresponding to cell types
            output_graph = construct_directed_network(adata, network_label, node_sep)

            # Plot cell type ligand pairs
            celltype_ligands = output_graph.nodes()
            celltypes = sorted(list(set([pair.split('_')[0].replace(celltype_sep_new, celltype_sep_old) for pair in celltype_ligands])))

            # Initialise the centres for each cluster of nodes (by cell type) around a big circle for 
            celltype_centres = {}
            num_subgraph_centres = len(celltypes)
                        
            for celltype in celltypes:
                celltype_index = celltypes.index(celltype)
                theta = 2.0 * np.pi * float(celltype_index) / float(num_subgraph_centres)
                celltype_centres[celltype] = np.array([circle_radius_outer * np.cos(theta), circle_radius_outer * np.sin(theta)])

            # Now calculate the positions for each graph node
            output_graph_pos = {}
            for celltype in celltypes:

                # Get the nodes corresponding to these clusters
                celltype_nodes = [n for n in output_graph.nodes() if celltype == n.split(node_sep)[0]]

                for node in celltype_nodes:
                    theta = 2.0 * np.pi * np.random.rand(1)[0]
                    radius = np.sqrt( circle_radius_inner**2.0 * np.random.rand(1)[0])

                    output_graph_pos[node] = np.array([radius*np.cos(theta), radius*np.sin(theta)]) + celltype_centres[celltype_centres]

                node_colours = [celltypes.index(node.split(node_sep)[0]) for node in output_graph.nodes()]
                node_labels = {node:node.split(node_sep)[1] for node in output_graph.nodes()}
                orig_edge_weights = list(nx.get_edge_attributes(output_graph,'weight').values())
                edge_weights = [edge_width_scale*weight for weight in orig_edge_weights]
                node_borders = [node_border_scale*output_graph[node] for node in output_graph.nodes()]

                edge_colours = [celltypes.index(edge[0].split(node_sep)[0]) for edge in output_graph.edges()]

                plt.figure(figsize=(12, 12)) 
                plt.rcParams["font.family"] = "Arial"
                plt.rcParams["font.size"] = font_size
                nodes = nx.draw_networkx_nodes(output_graph, output_graph_pos, node_color=node_colours, cmap=node_color_map, node_size=50, edgecolors='black', linewidths=node_borders)
                edges = nx.draw_networkx_edges(
                    output_graph,
                    output_graph_pos,
                    edge_cmap=edge_color_map,
                    edge_color=edge_colours,
                    connectionstyle="arc3,rad=0.1",
                    width=list(edge_weights), 
                    arrowsize=arrow_size,
                    alpha=edge_alpha)
                plt.axis('off')

                if label_nodes:
                    nx.draw_networkx_labels(output_graph, output_graph, node_labels, font_family='Arial')

                if output_name:
                    cwd = os.getcwd() + '/'
                    plt.savefig(cwd + output_name + '.pdf', transparent=True, bbox_inches='tight')
                plt.show()

def plot_multipartite_network(adata: anndata.AnnData, 
                            network_label: str = 'causal_networks', 
                            output_name: Optional[str] = None,
                            label_nodes: bool = True, 
                            celltype_sep_old: str =' ',
                            celltype_sep_new: str = '-',
                            node_sep: str = '_',
                            edge_type: str = 'condition',
                            partite_scale: float = 1.5,
                            ligand_label_shift: float = 1.1,
                            celltype_label_shift: float = 1.05,
                            node_size: int = 50,
                            node_border_scale: float = 2.0,
                            node_color_map: str = 'tab10',
                            edge_color_map: str = 'Set3',
                            edge_width_scale: float = 1.0,
                            arrow_size: int = 15,
                            edge_alpha: float = 0.6,
                            font_size: int = 12, 
                            xmargin: float = 0.2,
                            ymargin: float = 0.1):

    if network_label not in ['base_networks', 'causal_networks']:

        print("Current network options only include 'base_networks' or 'causal_networks'.")

    else:

        if edge_type not in ['condition', 'source', 'target']:

            print('Edge type not supported. Must be one of: source, target, condition')

        else:

            # Construct the directed graph where nodes are marked by 'layers' corresponding to cell types
            output_graph = construct_directed_network(adata, network_label, node_sep)

            # Plot cell type ligand pairs
            celltype_ligands = output_graph.nodes()
            celltypes = sorted(list(set([pair.split('_')[0].replace(celltype_sep_new, celltype_sep_old) for pair in celltype_ligands])))

            # Plot the full PDAG for cell-type-specific ligands
            pos = nx.multipartite_layout(output_graph, subset_key='layer', scale=partite_scale)

            # Translate the labels to give more space between partitions
            x_coords = np.sort(np.unique([pos[node][0] for node in pos]))
            x_min = x_coords.min()
            y_max = max([pos[node][1] for node in pos])

            # Ligand labels are on the left hand side of the network
            node_labels = {node:node.split(node_sep)[1] for node in output_graph.nodes()}
            for node in output_graph.nodes():
                if (pos[node][0] > x_min):
                    node_labels[node] = ''

            node_labels_pos = {}

            for node in pos:
                coords = pos[node].copy()

                if node_labels[node] != '':
                    coords[0] *= ligand_label_shift

                node_labels_pos[node] = coords

            node_colors = [node_color_map[celltypes.index(node.split(node_sep)[0])] for node in output_graph.nodes()]
            labels = {node:node for node in output_graph.nodes()}
            edge_colors = [edge_color_map[output_graph.edges[edge[0], edge[1]][edge_type]] for edge in output_graph.edges(data=True)]
            orig_edge_weights = list(nx.get_edge_attributes(output_graph,'weight').values())
            edge_weights = [edge_width_scale*weight for weight in orig_edge_weights]

            # Construct the dictionary of perturbed targets to colour in node borders
            perturbed_targets_dict = {node:0.0 for node in output_graph.nodes()}

            if network_label == 'causal_networks':
                perturbed_targets_list = adata.uns[network_label]['perturbed_targets']

                for targets_list in perturbed_targets_list:
                    for i in range(len(targets_list)):
                        node = celltype_ligands[i]

                        if node in perturbed_targets_dict:
                            perturbed_targets_dict[node] += targets_list[i]
                        else:
                            perturbed_targets_dict[node] = 0.0

                for node in perturbed_targets_dict: # Average over the intervened frequencies to deal with multiple perturbations
                    perturbed_targets_dict[node] /= (1.0 * len(perturbed_targets_list))

            node_borders = [node_border_scale * perturbed_targets_dict[node] for node in output_graph.nodes()]

            plt.figure(figsize=(15, 12)) 
            plt.rcParams["font.family"] = "Arial"
            plt.rcParams["font.size"] = font_size
            nodes = nx.draw_networkx_nodes(output_graph, pos, node_color=node_colors, cmap=plt.get_cmap(node_color_map), node_size=node_size, edgecolors='black', linewidths=node_borders)
            edges = nx.draw_networkx_edges(output_graph,
                                            pos,
                                            edge_cmap=plt.get_cmap(edge_color_map),
                                            edge_color=edge_colors,
                                            connectionstyle="arc3,rad=0.1",
                                            width=list(edge_weights), 
                                            arrowsize=arrow_size,
                                            alpha=edge_alpha)

            if label_nodes:
                nx.draw_networkx_labels(output_graph, node_labels_pos, node_labels, font_family='Arial')

                # Add the cluster labels to the top as well
                for celltype in celltypes:
                    celltype_index = celltypes.index(celltype)
                    celltype_xpos = x_coords[celltype_index]
                    plt.text(celltype_xpos, celltype_label_shift*y_max, celltype.replace(celltype_sep_new, celltype_sep_old), ha='left', rotation=45)

            plt.margins(x=xmargin)
            plt.margins(y=ymargin)
            plt.axis('off')

            if output_name:
                cwd = os.getcwd() + '/'
                plt.savefig(cwd + output_name + '.pdf', transparent=True, bbox_inches='tight')

            plt.show()

def plot_spatial_regions(adata: anndata.AnnData,
                        coord_labels: List[str], 
                        celltype_label: str, 
                        condition_label: str,
                        output_name: Optional[str] = None,
                        network_label: str = 'causal_network',
                        jiggle_scale: float = 1.0,
                        node_cmap: str = 'tab10'):

    conditions = adata.obs[condition_label].unique().tolist()
    celltype_ligands = list(adata.uns[network_label]['celltype_ligands'])

    # Get the feasible pairs. We can't run this method without it.
    feasible_pairs = adata.uns[network_label]['feasible_pairs']

    # We will plot the medoid for each cell type
    coord_medoids = {condition:{} for condition in conditions}
    for condition in coord_medoids:
        adata_subset = adata[adata.obs[condition_label] == condition]
        adata_subset.obs[celltype_label].cat.remove_unused_categories(inplace=True) # To prevent weird behaviour

        for celltype in adata_subset.obs[celltype_label].unique():
            adata_for_celltype = adata_subset[adata_subset.obs[celltype_label] == celltype]
            spatial_coords = adata_for_celltype.obs[coord_labels].to_numpy()
            pairwise_distances = distance_matrix(spatial_coords, spatial_coords)
            medoid_index = np.argmin(pairwise_distances.sum(axis=0))
            medoid = spatial_coords[medoid_index, :]
            angle = 2.0*np.pi*np.random.random(1)
            coord_medoids[condition][celltype] = medoid + jiggle_scale*np.array([np.cos(angle), np.sin(angle)]).flatten()

    celltype_transitions = {} # Directed flows
    cellproximity_adjacencies = {} # Undirected spatial adjacencies

    for condition in conditions:
        adata_subset = adata[adata.obs[condition_label] == condition]
        celltypes = sorted(adata_subset.obs[celltype_label].unique().tolist())
        celltype_transitions[condition] = np.zeros((len(celltypes), len(celltypes)))
        cellproximity_adjacencies[condition] = np.zeros((len(celltypes), len(celltypes)))

    for condition in cellproximity_adjacencies:
        transitions = celltype_transitions[condition]
        cellproximities = cellproximity_adjacencies[condition]
        adata_subset = adata[adata.obs.Group == condition]
        adata.obs.AT.cat.remove_unused_categories(inplace=True)


#     for index, row in cellproximities.iterrows():

#         cluster_A = row['cell_1']
#         cluster_B = row['cell_2']
#         cluster_A_index = clusters.index(cluster_A)
#         cluster_B_index = clusters.index(cluster_B)

        
#         adjacencies[cluster_A_index, cluster_B_index] = row['enrichm'] # Add the adjacency if it's found from Giotto

#     cellproximity_adjacencies[condition] = adjacencies

#     # Define the connectivity graph
#     connectivity_graph = nx.Graph()
#     connectivity_graph.add_nodes_from(clusters)

#     nonzero_rows, nonzero_cols = adjacencies.nonzero()

#     for j in range(len(nonzero_rows)):

#         node_1 = clusters[nonzero_rows[j]]
#         node_2 = clusters[nonzero_cols[j]]

#         if node_1 != node_2:
#             connectivity_graph.add_edge(node_1, node_2)

#     # Plot the giotto networks on top of the scatter plots
#         # Create plot

#     sc.pl.scatter(adata, x=coord_labels[0], y=coord_labels[1], color=celltype_label,
#                 legend_loc='none', size=100.0, title=condition, alpha=0.25,
#                 frameon=False, show=False)

#     # Plot the network now
#     edge_colors = [node_color_map[clusters.index(edge[0])] for edge in connectivity_graph.edges()]
#     node_colors = [node_color_map[clusters.index(node)] for node in connectivity_graph.nodes()]
#     nodes = nx.draw_networkx_nodes(connectivity_graph, coord_medoids[condition], node_color=node_colors, node_size=75.0)
#     edges = nx.draw_networkx_edges(
#         connectivity_graph,
#         coord_medoids[condition],
#         edge_color='black',
#         connectionstyle="arc3,rad=0.2",
#         alpha=0.5,
#         width=1.0,
#     )

#     plt.axis('off')
#     plt.savefig(data_directory + 'chen20_3mo_cellproximity_scatter_' + condition + '.pdf')

#     causality_flow_graph = nx.DiGraph()

#     causality_flow_graph.add_nodes_from(clusters)

#     # We will load the original inferred graph
#     # cluster_ligand_graph = nx.read_edgelist(data_directory + 'chen20_3mo_cccflow_utigsp_adjacency_parcorr_bagged_' + condition + '_edgetype.edgelist.gz',
#     #                                                 create_using=nx.DiGraph(), data=[('weight', float)])

#     inferred_edges = pd.read_csv(data_directory + 'chen20_3mo_cccflow_utigsp_adjacency_parcorr_bagged_ligand_target_edges.csv')
#     inferred_edges_subset = inferred_edges[(inferred_edges['Group'] == condition)|(inferred_edges['Group'] == 'Both') ]
#     intervention_targets = np.genfromtxt(data_directory + "chen20_3mo_celltype_utigsp_interventiontargets_parcorr_bagged.csv", delimiter="\n")

#     total_weight = 0
#     for index, row in inferred_edges_subset.iterrows():

#         node_1_cluster = row['Source']
#         node_2_cluster = row['Target']

#         cluster_1_index = clusters.index(node_1_cluster)
#         cluster_2_index = clusters.index(node_2_cluster)

#         edge_weight = row['Frequency']

#         transitions[cluster_1_index, cluster_2_index] += edge_weight
#         total_weight += edge_weight
#     cluster_transitions[condition] = transitions

#     intervened_targets_dict = {cluster:0.0 for cluster in clusters}
#     for i in range(len(intervention_targets)):
#         cluster_ligand = cluster_ligands[i]
#         cluster = cluster_ligand.split('_')[0]
#         intervention_target_freq = intervention_targets[i]

#         if cluster in clusters:
#             cluster_index = clusters.index(cluster)

#             if ( (transitions[cluster_index,:].sum() != 0)|(transitions[:, cluster_index].sum() != 0) ):
#                 if cluster in intervened_targets_dict:
#                     intervened_targets_dict[cluster] += intervention_target_freq
#                 else:
#                     intervened_targets_dict[cluster] = intervention_target_freq



#     nonzero_rows, nonzero_cols = transitions.nonzero()

#     for j in range(len(nonzero_rows)):

#         node_1 = clusters[nonzero_rows[j]]
#         node_2 = clusters[nonzero_cols[j]]

#         causality_flow_graph.add_edge(node_1, node_2)
#         causality_flow_graph.edges[node_1, node_2]['weight'] = transitions[nonzero_rows[j], nonzero_cols[j]]

#         # Plot the giotto networks on top of the scatter plots
#         # Create plot
#     plt.figure(figsize=(6, 6))

#     sc.pl.scatter(adata, x='coord_X', y='coord_Y', color='AT',
#                 legend_loc='none', size=100.0, title=condition, alpha=0.1,
#                 frameon=False, show=False)

#     # Plot the network now
#     edge_colors = [node_color_map[celltypes.index(edge[0])] for edge in causality_flow_graph.edges()]
#     node_colors = [node_color_map[celltypes.index(node)] for node in causality_flow_graph.nodes()]
#     orig_edge_weights = list(nx.get_edge_attributes(causality_flow_graph,'weight').values())
#     edge_weights = [edge_width_scale*weight/ total_weight for weight in orig_edge_weights]
#     node_borders = [node_border_scale*intervened_targets_dict[node] for node in causality_flow_graph.nodes()]

#     nodes = nx.draw_networkx_nodes(causality_flow_graph, coord_medoids[condition], node_color=node_colors, node_size=node_size, edgecolors='black', linewidths=node_borders)
#     edges = nx.draw_networkx_edges(
#         causality_flow_graph,   
#         coord_medoids[condition],
#         edge_color=edge_colors,
#         connectionstyle="arc3,rad=0.5",
#         alpha=1.0,
#         width=edge_weights,
#     )

#     plt.axis('off')
#     plt.savefig(data_directory + 'chen20_3mo_causalflows_scatter_' + condition + '.pdf')



