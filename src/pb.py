"""Calculate pseudobulk differential expression for Xenium data."""

import os
from pathlib import Path

import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.sparse import issparse

from utils.seed_everything import seed_everything
from utils.setup_logger import setup_logger


# Functions
def pseudobulk_sampleID_celltype(adata, celltype_col=None, donor_col="ROI", agg="sum"):
    """Create a pseudobulk matrix: one column per (donor, cell_type) combination.

    Parameters:
        adata: AnnData object
        celltype_col: obs column with cell type labels
        donor_col: obs column with donor / ROI labels
        agg: aggregation method — "sum" (default) or "mean"

    Returns:
        pb: pd.DataFrame  shape (n_genes, n_groups)
            columns named "Donor1__T cells", "Donor1__Macrophages", etc.
        meta: pd.DataFrame  shape (n_groups, n_meta_cols)
            columns named "Donor1__T cells", donor, cell_type, n_cells
    """
    assert agg in ("sum", "mean"), "agg must be 'sum' or 'mean'"

    # Pull expression into a dense DataFrame (cells x genes)
    X = adata.layers["counts"]  # use raw counts for pseudobulk
    if issparse(X):
        X = X.toarray()

    df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    df[donor_col] = adata.obs[donor_col].values
    df[celltype_col] = adata.obs[celltype_col].values

    # Group and aggregate
    grouped = df.groupby([donor_col, celltype_col])

    pb = grouped.sum() if agg == "sum" else grouped.mean()  # (n_groups, n_genes)

    # Transpose: rows = genes, columns = (donor, cell_type)
    pb = pb.T

    # Flatten MultiIndex columns → "Donor1__T cells"
    pb.columns = [f"{donor}__{ct}" for donor, ct in pb.columns]

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

    # Create a combined "ROI_celltype" column for indexing
    meta["ROI_celltype"] = meta.apply(
        lambda r: f"{r[donor_col]}__{r[celltype_col]}", axis=1
    )
    # Set the index to "ROI_celltype" for easier joining later
    meta.set_index("ROI_celltype", inplace=True)

    # Confirm ROI_celltype is a string type
    meta.index = meta.index.astype("string")

    return pb, meta


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


# Set variables
level = "level_2"
ROI_names = "ROI"


# Set directory
path = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium"
)
dir = path / "output/2026-03-27_analysis_run"
input_dir = path / "data"

# Set figure directory
folder_name = "pb_data"
out_path = f"project_analysis/general/{folder_name}"

out_dir = dir / out_path
os.makedirs(out_dir, exist_ok=True)

# set fig dir for plots to save to
sc.settings.figdir = out_dir

# Set colors
cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
color_palette_level_1 = sns.color_palette("hls", 12)

# Load data
logger.info("Loading data...")
adata = sc.read_h5ad(dir / "annotate/adata_level_2_level_3.h5ad")  # full dataset

logger.info(f"adata shape: {adata.shape}")
logger.info(f"adata.obs columns: {adata.obs.columns.tolist()}")
logger.info(adata)


# Add in additional metadata columns for downstream analysiss
# Extract relevant metadata from adata.obs
logger.info("Extracting metadata from adata.obs...")
adata_meta = (
    adata.obs[
        [ROI_names, "sample_ID", "batch", "condition", "timepoint", "timepoint_label"]
    ]
    .drop_duplicates(subset=[ROI_names])
    .set_index(ROI_names)
)

# Add metadata from excel file
logger.info("Loading manual metadata from Excel file...")
manual_meta = pd.read_csv(
    path / "data/meta/STx_meta_analysis_only_cleaned.csv", index_col="ROI"
)

### CALCULATE PSEUDOBULK AND PCA ON DONOR AND CELL LEVEL

# Pseudobulk and PCA on all samples
logger.info("Calculating pseudobulk for each cell type for each donor...")
pb, meta = pseudobulk_sampleID_celltype(
    adata, celltype_col=level, donor_col=ROI_names, agg="sum"
)

# Filter to remove groups with too few cells (e.g. <10) to avoid noisy DE results
min_cells = 10
logger.info(f"Keeping groups with at least {min_cells} cells")

dropped = meta[meta["n_cells"] < min_cells].apply(
    lambda r: f"{r[ROI_names]}__{r[level]}", axis=1
)
logger.info(f"Dropping {len(dropped)} groups with fewer than {min_cells} cells")
logger.info("Dropped groups:\n%s", "\n".join(dropped))

# Filter pseudobulk columns (samples), not rows (genes)
keep_cols = [col for col in pb.columns if col not in dropped.values]
pb_filtered = pb[keep_cols]

# Filter metadata to match pseudobulk matrix
meta_filtered = meta.loc[keep_cols]

# Combine into one meta data df
meta_df = meta_filtered.merge(
    adata_meta,
    left_on=ROI_names,
    right_index=True,
    how="left",
    suffixes=("", "_adata_meta"),
)

meta_df = meta_df.merge(
    manual_meta,
    left_on=ROI_names,
    right_index=True,
    how="left",
    suffixes=("", "_manual"),
)

logger.info(f"Filtered pseudobulk matrix shape: {pb_filtered.shape}")
logger.info(f"Filtered metadata shape: {meta_df.shape}")

# Save pseudobulk matrix and metadata
pb_filtered.to_csv(out_dir / "pseudobulk_matrix_ROI_celltype.csv", index=True)
meta_df.to_csv(out_dir / "pseudobulk_metadata_ROI_celltype.csv", index=True)


### CALCULATE PSEUDOBULK AND PCA ON DONOR LEVEL (NOT SPLIT BY CELL TYPE)
logger.info("Calculating pseudobulk for each donor...")
pb_sample, meta_sample = pseudobulk_sampleID(adata, donor_col=ROI_names, agg="sum")

# Combine into one meta data df
meta_df = meta_sample.merge(
    adata_meta,
    left_on=ROI_names,
    right_index=True,
    how="left",
    suffixes=("", "_adata_meta"),
)

meta_df = meta_df.merge(
    manual_meta,
    left_on=ROI_names,
    right_index=True,
    how="left",
    suffixes=("", "_manual"),
)

logger.info(f"Pseudobulk matrix shape: {pb_sample.shape}")
logger.info(f"Metadata shape: {meta_df.shape}")

# Save pseudobulk matrix and metadata
pb_sample.to_csv(out_dir / "pseudobulk_matrix_ROI.csv", index=True)
meta_df.to_csv(out_dir / "pseudobulk_metadata_ROI.csv", index=True)

logger.info("Pseudobulk matrix and metadata saved.")
