"""Add metadata from JSON file to adata.obs based on ROI mapping."""

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch


# Functions
def seed_everything(seed: int):
    """Set random seed on every random module for reproducibility.

    Args:
        seed: The seed value to set for random number generation.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    else:
        pass


def setup_logger(log_dir: Path, log_name: str) -> logging.Logger:
    """Set up a logger that writes to both console and a timestamped file.

    Args:
        log_dir: Directory where the log file will be saved.
        log_name: Base name for the log file.

    Returns:
        Configured logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{log_name}_{timestamp}.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if function is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger


# Set random seed for reproducibility
seed_everything(19960915)


# Set up logger
wd = "/rds/general/user/sep22/home/Projects/AirScape/HPC_jobs/rejuvenair/"
logs_dir = Path(wd) / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir=logs_dir, log_name="add_meta")
logger.info("Logger set up successfully. Logs will be saved to: %s", logs_dir)

# Set directory
dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
)

# Set path to adata directory
adata_dir = dir / "output/2026-03-27_analysis_run/"

# Set path to JSON metadata file
json_path = dir / "data/meta/meta_rejuvenair.json"

# Load adata
adata = sc.read_h5ad(adata_dir / "subset_adata/adata_subset_COPD_MICA.h5ad")

# Open json
with open(json_path) as f:
    metadata_dict = json.load(f)

# Convert to DataFrame
metadata_df = pd.DataFrame.from_dict(metadata_dict, orient="index")

logger.info("Metadata columns: list(metadata_df.columns)")
logger.info(f"Loaded metadata for {len(metadata_df)} samples")

# Map metadata to adata.obs based on ROI
# The JSON keys (e.g., "COPD_46005_V1") should `match your ROI column
for column in metadata_df.columns:
    adata.obs[column] = adata.obs["ROI"].astype(str).map(metadata_df[column])

# Check results
logger.info("\nAdded columns to adata.obs:")
for col in metadata_df.columns:
    n_missing = adata.obs[col].isna().sum()
    n_unique = adata.obs[col].nunique()
    logger.info(f"  {col}: {n_unique} unique values, {n_missing} missing")

# Verify mapping
logger.info("\nSample of mapped data:")
logger.info(adata.obs[["ROI"] + list(metadata_df.columns)].head(10))

# Before
keys_before = set(adata.obs.columns)

for column in metadata_df.columns:
    adata.obs[column] = adata.obs["ROI"].astype(str).map(metadata_df[column])

# After - confirm only obs.columns changed
keys_after = set(adata.obs.columns)
logger.info("New columns added: %s", keys_after - keys_before)
logger.info("uns intact: %s", list(adata.uns.keys()))
logger.info("obsm intact: %s", list(adata.obsm.keys()))
logger.info("layers intact: %s", list(adata.layers.keys()))

# Add a column for treatment and timepoint
adata.obs["time_and_treatment_arm"] = (
    adata.obs["timepoint_meta"] + "_" + adata.obs["treatment_arm"].astype(str)
)
logger.info("\nAdded 'time_and_treatment_arm' column to adata.obs.")
logger.info(adata.obs["time_and_treatment_arm"].value_counts())

# Add column for collected level 1 labels
logger.info("Rename level_1 categories for clarity...")
immune_cell_types = [
    "Mast cells",
    "Monocytes",
    "Lipid-associated macrophages",
    "Dendritic cells",
    "Mast cells",
    "CD4+ T cells",
    "Dendritic cells",
    "Plasma cells",
    "Macrophages",
    "T cells",
    "Interstitial Macrophages",
    "Plasma cells",
    "CD8+ T cells",
    "B cells",
    "Plasma cells",
    "Neutrophils",
    "NK cells",
    "CD4+ T cells",
    "Interstitial Macrophages",
]

airway_cell_types = [
    "Ciliated cells",
    "Goblet cells",
    "Basal cells",
    "Secretory epithelial cells",
    "Proliferating Basal cells",
]

endothelial_cell_types = [
    "Blood endothelial cells - unclassified",
    "Capillary endothelial cells",
    "Pulmonary artery endothelial cell",
    "Pulmonary vein endothelial cell",
    "Lymphatic endothelial cells",
    "Aerocytes",
]

stromal_cell_types = [
    "SMC",
    "Adventitial fibroblasts",
    "Pericytes",
    "CTHRC1+ fibroblasts",
    "Adventitial fibroblasts",
    "Alveolar fibroblasts",
    "Lipo-fibroblasts",
]

adata.obs["level_1_corrected"] = adata.obs["level_2"].map(
    lambda x: "Immune cells"
    if x in immune_cell_types
    else "Airway epithelial cells"
    if x in airway_cell_types
    else "Endothelial cells"
    if x in endothelial_cell_types
    else "Stromal cells"
    if x in stromal_cell_types
    else "Other"
)

# LOOKING AT THE UNIQUE CATEGORIES IN THE LEVEL_1 COLUMN
level_1_list = adata.obs["level_1"].unique().tolist()
logger.info(f"Unique level_1 categories: {level_1_list}")
# Look at the number of level_2 cells in each level_1 category
for level_1 in level_1_list:
    n_cells = adata.obs[adata.obs["level_1"] == level_1].shape[0]
    logger.info(f"Number of cells in level_1 category '{level_1}': {n_cells}")
    logger.info(
        f"Unique level_2 categories in level_1 category '{level_1}': "
        f"{adata.obs[adata.obs['level_1'] == level_1]['level_2'].value_counts()}"
    )

# LOOKING AT THE UNIQUE CATEGORIES IN THE LEVEL_1 COLUMN
level_1_list = adata.obs["level_1_corrected"].unique().tolist()
logger.info(f"Unique level_1_corrected categories: {level_1_list}")
# Look at the number of level_2 cells in each level_1 category
for level_1 in level_1_list:
    n_cells = adata.obs[adata.obs["level_1_corrected"] == level_1].shape[0]
    logger.info(f"Number of cells in level_1 category '{level_1}': {n_cells}")
    logger.info(
        f"Unique level_2 categories in level_1 category '{level_1}': "
        f"{adata.obs[adata.obs['level_1_corrected'] == level_1]['level_2'].value_counts()}"
    )

# Save updated adata
adata.write_h5ad(adata_dir / "subset_adata/adata_subset_COPD_MICA_meta.h5ad")
logger.info("Adding metadata completed and saved successfully.")
