"""Remove small clusters with fewer than 10 cells."""

def remove_small_clusters(celltype_level, adata, num_cells_threshold=10):
    """Remove clusters with fewer than a specified number of cells.
    Args:
        celltype_level: The column name in adata.obs that contains the cluster labels.
        adata: The AnnData object containing the data.
        num_cells_threshold: The minimum number of cells required for a cluster to be retained (default is 10).
    Returns:
        adata_clean: The filtered AnnData object with small clusters removed.
    """
    # Remove cells with NaN celltype_level labels
    n_before = adata.n_obs
    adata_clean = adata[adata.obs[celltype_level].notna()].copy()

    # Filter adata to include only clusters with at least num_cells_threshold cells
    adata_clean_filtered = adata_clean[
        adata_clean.obs[celltype_level].isin(
            adata_clean.obs[celltype_level]
            .value_counts()[adata_clean.obs[celltype_level].value_counts() >= num_cells_threshold]
            .index
        )
    ].copy()

    return adata_clean_filtered
