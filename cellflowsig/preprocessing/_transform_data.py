import numpy as np
import itertools
import anndata as ad

def construct_celltype_ligand_expressions(adata, celltype_label, celltype_sep_old=' ', celltype_sep_new='-', node_sep='_', expressions_label='X_celltype_ligand', base_network_label='base_networks'):

    # Check if the base networks hvae been constructed
    if base_network_label not in adata.uns:
        print('You need to construct the base network from cell-cell communication output before constructing cell-type-ligand expression.')
        
    else:

        celltype_ligand_pairs = adata.uns[base_network_label]['celltype_ligands']

        celltype_ligand_expressions =  np.zeros((adata.n_obs, len(celltype_ligand_pairs))) # Initialise the matrix

        for i in range(len(celltype_ligand_pairs)):
            celltype_ligand = celltype_ligand_pairs[i].split(node_sep)
            
            celltype = celltype_ligand[0].replace(celltype_sep_new, celltype_sep_old)
            ligand = celltype_ligand[1] 
            
            celltype_indices = np.where(adata.obs[celltype_label] == celltype)[0]
                    
            interaction_expression = adata[celltype_indices, ligand]
                    
            celltype_ligand_expressions[celltype_indices, i] = interaction_expression.reshape((len(celltype_indices), ))

        # Construct annotated dataframe 
        adata.obsm[expressions_label] = celltype_ligand_expressions
