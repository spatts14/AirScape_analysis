"""Subset adata based on metadata columns."""

# Subset adata based on metadata columns
import logging
import os
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc


def subset_adata_by_meta(
    adata: ad.AnnData,
    meta_col: str,
    meta_values: list[str],
    logger=None,
) -> ad.AnnData:
    """Subset an AnnData object based on values in a metadata column.

    Args:
    adata : anndata.AnnData
        Input AnnData object.
    meta_col : str
        Column name in adata.obs to subset on.
    meta_values : list[str]
        Values in meta_col to retain.
    logger : logging.Logger, optional
        Logger for error reporting.

    Returns:
    anndata.AnnData
        Subsetted AnnData object containing only cells matching meta_values.
    """
    # Check column exists
    if meta_col not in adata.obs.columns:
        msg = f"Meta column '{meta_col}' not found in adata.obs.columns"
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    # Check values exist
    available_values = set(adata.obs[meta_col].unique())
    missing_values = [val for val in meta_values if val not in available_values]

    if missing_values:
        msg = f"Meta values {missing_values} not found in meta column '{meta_col}'"
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    # Subset
    mask = adata.obs[meta_col].isin(meta_values)
    adata_subset = adata[mask].copy()

    return adata_subset


def subset_random_rois(
    adata: ad.AnnData,
    roi_col: str,
    n_rois: int = 3,
    random_state: int = 42,
    logger=None,
) -> ad.AnnData:
    """Randomly subset AnnData to a fixed number of ROIs.

    Args:
        adata:
            Input AnnData object.
        roi_col:
            Column in adata.obs containing ROI labels.
        n_rois:
            Number of ROIs to sample.
        random_state:
            Seed for reproducibility.
        logger:
            Optional logger.

    Returns:
        Subsetted AnnData object.
    """
    if roi_col not in adata.obs.columns:
        msg = f"ROI column '{roi_col}' not found in adata.obs"
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    unique_rois = adata.obs[roi_col].unique()

    if len(unique_rois) < n_rois:
        msg = f"Requested {n_rois} ROIs, but only {len(unique_rois)} available."
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    rng = np.random.default_rng(random_state)

    selected_rois = rng.choice(
        unique_rois,
        size=n_rois,
        replace=False,
    )

    if logger:
        logger.info(f"Selected ROIs: {selected_rois}")

    mask = adata.obs[roi_col].isin(selected_rois)

    return adata[mask].copy()


# Set directories and file names
base_dir = Path(os.getenv("BASE_DIR"))
module_dir = base_dir / "annotate"

output_dir = base_dir / "subset_adata"
os.makedirs(output_dir, exist_ok=True)

# Set up logging
log_file = output_dir / "subsetting_by_meta.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Set variables
h5ad_file = "adata_level_2_level_3.h5ad"  # name of annotated adata file to subset
condition_values = ["COPD"]  # conditions to keep
timepoint_values = ["V1", "V2", "V3"]  # timepoint to keep
file_name = "COPD.h5ad"  # name of subsetted adata file to save

condition_col = "condition"
timepoint_col = "timepoint"

# Load annotated adata
logger.info(f"Loading data from {module_dir / f'{h5ad_file}'}")
adata = sc.read_h5ad(module_dir / f"{h5ad_file}")
logger.info(f"Data loaded successfully. Shape: {adata.shape}")

# Step 1: subset by condition
logger.info(
    f"Subsetting data based on column '{condition_col}' and values {condition_values}"
)
adata_condition_subset = subset_adata_by_meta(
    adata,
    meta_col=condition_col,
    meta_values=condition_values,
    logger=logger,
)
logger.info(f"Condition subset complete. New shape: {adata_condition_subset.shape}")
logger.info(
    f"Conditions in subset: {adata_condition_subset.obs[condition_col].value_counts()}"
)
logger.info(
    f"Timepoints in subset: {adata_condition_subset.obs[timepoint_col].value_counts()}"
)

# Step 2: subset the condition-filtered data by timepoint

logger.info(
    "Subsetting condition-filtered data based on column "
    f"'{timepoint_col}' and values {timepoint_values}"
)
adata_subset = subset_adata_by_meta(
    adata_condition_subset,
    meta_col=timepoint_col,
    meta_values=timepoint_values,
    logger=logger,
)
logger.info(f"Final subset complete. New shape: {adata_subset.shape}")

# Select ROIs to keep at random for toy dataset (to reduce size for testing)
# roi_col = "ROI"
# adata_subset = subset_random_rois(
#     adata_subset,
#     roi_col=roi_col,
#     n_rois=4,
#     random_state=42,
#     logger=logger,
# )

# logger.info(f"ROI subset complete. New shape: {adata_subset.shape}")
# logger.info(f"ROIs in subset: {adata_subset.obs[roi_col].value_counts()}")

# Check output of subsetted adata
logger.info(f"Condition subset complete. New shape: {adata_subset.shape}")
logger.info(f"Conditions in subset: {adata_subset.obs[condition_col].value_counts()}")
logger.info(f"Timepoints in subset: {adata_subset.obs[timepoint_col].value_counts()}")

# Save new adata
logger.info(f"Saving subsetted data to {output_dir / file_name}")
adata_subset.write_h5ad(output_dir / file_name)
logger.info("Subsetted data saved successfully.")
