import numpy as np
import itertools
import anndata

def construct_celltype_ligand_expressions(adata:anndata.AnnData,\
                                        celltype_label: str,
                                        celltype_sep_old: str = ' ',
                                        celltype_sep_new: str = '-',
                                        node_sep: str = '_',
                                        expressions_label: str = 'X_celltype_ligand',
                                        base_network_label: str = 'base_networks'):
    """
    Constructs a N x K transformed version of the original N x M gene expression matrix,
    where N is the number of cells, M is the number of original genes, and K is the number
    of cell-type-ligand pairs implicated in causal activity by the constructed base network
    from cell-cell communication inference.
    
    We construct a new expression matrix, Y, such that for every ligand, $L$, in the base network,
    $Y_{(C, L)} = X_{iL}$ if cell $i$ annotated by cell type C and $Y_{(C, L)} = 0$ otherwise.

    Parameters
    ----------
    adata
        Annotated dataframe of original gene expression matrix, with cell type label and base network
        that has already been constructed.

    celltype_label
        Cell type label used to construct cell-type-ligand expression. Must correspond to cell type
        label used to create base network.

    celltype_sep_old
        Original string used to separate cell type labels, typically ' ' or '_'. Replaced by typically
        '-'.

    celltype_sep_new
        String used to reformat cell type labels for later use. Typically '-' is used to replace
        spaces or underlines.

    node_sep
        String used to concatenate cell type and ligand pairs.

    expressions_label
        The label that is given to the constructed cell-type-ligand expression matrix in adata.obsm.

    base_network_label
        Where the information about the base network have been stored.

    Returns
    -------
    celltype_ligand_expressions
        The transformed cell-type-ligand expression matrix, stored in adata.obsm[expressions_label].
    
    """
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
