"""Calculate pseudobulk and PCA for each cell type in Xenium data."""

import os
from pathlib import Path

import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

from utils.safe_name import safe_name
from utils.seed_everything import seed_everything
from utils.setup_logger import setup_logger


# Functions
def pseudobulk_sampleID(adata, donor_col="ROI", agg="sum"):
    """Create a pseudobulk matrix: one column per (donor, cell_type) combination.

    Parameters:
        adata: AnnData object
        donor_col: obs column with donor / ROI labels
        agg: aggregation method — "sum" (default) or "mean"

    Returns:
        pb: pd.DataFrame  shape (n_genes, n_groups)
            columns named "Donor1__T cells", "Donor1__Macrophages", etc.
        meta: pd.DataFrame  shape (n_groups, n_meta_cols)
            columns named "Donor1__T cells", donor, cell_type, n_cells
    """
    assert agg in ("sum", "mean"), "agg must be 'sum' or 'mean'"

    X = adata.layers["counts"]
    if issparse(X):
        X = X.toarray()

    df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    df[donor_col] = adata.obs[donor_col].values

    # Group and aggregate
    grouped = df.groupby(donor_col)
    pb = grouped.sum() if agg == "sum" else grouped.mean()  # (n_donors, n_genes)

    # Transpose: rows = genes, columns = donors
    pb = pb.T

    # Cell counts per group
    n_cells = grouped.size().rename("n_cells")

    # Calculate total transcripts per group
    gene_cols = adata.var_names
    total_transcripts = grouped[gene_cols].sum().sum(axis=1).rename("total_counts")

    # Calculate mean transcripts per cell in each group
    mean_transcripts = total_transcripts / n_cells
    if mean_transcripts.isna().any():
        # If there are any NaN values (e.g. due to division by zero)
        # Fill them with zero
        mean_transcripts = mean_transcripts.fillna(0)

    # Cell counts per group
    n_cells = grouped.size().rename("n_cells")

    # Calculate total transcripts per group
    gene_cols = adata.var_names
    total_transcripts = grouped[gene_cols].sum().sum(axis=1).rename("total_counts")

    # Calculate mean transcripts per cell in each group
    mean_transcripts = total_transcripts / n_cells
    if mean_transcripts.isna().any():
        # If there are any NaN values (e.g. due to division by zero)
        # Fill them with zero
        mean_transcripts = mean_transcripts.fillna(0)

    # Make metadata table
    meta = pd.DataFrame(
        {
            "n_cells": n_cells,
            "total_counts": total_transcripts,
            "mean_transcripts": mean_transcripts,
        }
    ).reset_index()

    # Set the index to donor_col for easier joining later
    meta.set_index(donor_col, inplace=True)

    # Ensure index is string type for downstream joinings
    meta.index = meta.index.astype("string")

    return pb, meta


# Set random seed for reproducibility
seed_everything(19960915)

# Set up logger
wd = "/rds/general/user/sep22/home/Projects/AirScape/HPC_jobs/general/"
logs_dir = Path(wd) / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir=logs_dir, log_name="pseudobulk")


# Set directory
path = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium"
)
dir = path / "output/2026-03-27_analysis_run"
input_dir = path / "data"

# Set figure directory
folder_name = "pb_data_celltype"
out_path = f"project_analysis/general/{folder_name}"

out_dir = dir / out_path
os.makedirs(out_dir, exist_ok=True)

# set fig dir for plots to save to
sc.settings.figdir = out_dir

# Set variables
level = "level_2"
ROI_names = "ROI"

# Load data
logger.info("Loading data...")
adata = sc.read_h5ad(dir / "annotate/adata_level_2_level_3.h5ad")

logger.info(f"adata shape: {adata.shape}")
logger.info(f"adata.obs columns: {adata.obs.columns.tolist()}")
logger.info(adata)

# Extract relevant metadata from adata.obs
logger.info("Extracting metadata from adata.obs...")
adata_meta = (
    adata.obs[
        [ROI_names, "sample_ID", "batch", "condition", "timepoint", "timepoint_label"]
    ]
    .drop_duplicates(subset=[ROI_names])
    .set_index(ROI_names)
)
logger.info(f"Extracted metadata: {adata_meta.head()}")

# Add metadata from excel file
logger.info("Loading manual metadata from Excel file...")
manual_meta = pd.read_csv(
    path / "data/meta/STx_meta_analysis_only_cleaned.csv", index_col="ROI"
)
logger.info(f"Manual metadata {manual_meta.head()}")

# Make empty dict
pseudo_celltype_dict = {}

# Make list of cell types to loop through
cell_types = adata.obs[level].cat.categories.tolist()
logger.info(f"Cell types: {cell_types}")

logger.info("Calculating pseudobulk for each cell type...")
for cell_type in cell_types:
    # Make directory for each cell type
    cell_type_safe = safe_name(
        cell_type
    )  # make safe name to use in directory and file names
    cell_type_dir = out_dir / cell_type_safe
    os.makedirs(cell_type_dir, exist_ok=True)

    logger.info(f"Calculating pseudobulk for cell type: {cell_type}...")
    # subset adata to cell type of interest
    adata_subset = adata[adata.obs[level] == cell_type].copy()

    logger.info("Calculating pseudobulk for each donor...")
    pb_sample, meta_sample = pseudobulk_sampleID(
        adata_subset, donor_col=ROI_names, agg="sum"
    )

    # Combine into one meta data df
    logger.info("Combining metadata from pseudobulk, adata.obs, and manual metadata...")
    meta_df = meta_sample.merge(
        adata_meta,
        left_on=ROI_names,
        right_index=True,
        how="left",
        suffixes=("", "_adata_meta"),
    )
    logger.info(f"Metadata after merging with adata.obs: {meta_df.head()}")

    meta_df = meta_df.merge(
        manual_meta,
        left_on=ROI_names,
        right_index=True,
        how="left",
        suffixes=("", "_manual"),
    )
    logger.info(f"Metadata after merging with manual metadata: {meta_df.head()}")

    # Remove samples with <n cells to reduce noise in PCA later on
    num_cells_threshold = 10
    logger.info(f"Filtering out samples with <{num_cells_threshold} cells...")
    valid_samples = meta_sample[meta_sample["n_cells"] >= num_cells_threshold].index
    pb_sample = pb_sample[valid_samples]
    meta_sample = meta_sample.loc[valid_samples]

    logger.info(f"{cell_type} filtered pseudobulk matrix shape: {pb_sample.shape}")
    logger.info(f"{cell_type} filtered metadata shape: {meta_sample.shape}")

    # Keep metadata aligned to the filtered pseudobulk matrix
    meta_df = meta_df.loc[valid_samples]

    # Save pseudobulk matrix and metadata
    pb_sample.to_csv(
        cell_type_dir / f"{cell_type_safe}_pseudobulk_matrix_ROI.csv", index=True
    )
    meta_df.to_csv(
        cell_type_dir / f"{cell_type_safe}_pseudobulk_metadata_ROI.csv", index=True
    )

    # Add to dict for later use in PCA module
    pseudo_celltype_dict[cell_type] = (pb_sample, meta_df)

logger.info("Pseudobulk matrix and metadata saved.")
