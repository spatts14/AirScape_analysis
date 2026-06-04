"""Clean adata to remove low quality of incorrect cell types."""

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch


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


# Set random seed for reproducibility
seed_everything(19960915)

# Set up logger
wd = "/rds/general/user/sep22/home/Projects/AirScape_analysis/HPC_jobs/rejuvenair/"
logs_dir = Path(wd) / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir=logs_dir, log_name="clean_adata")
logger.info("Logger set up successfully. Logs will be saved to: %s", logs_dir)

# Set directory

dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/2026-03-27_analysis_run/subset_adata"
)

# Load data
logger.info("Loading data...")
adata = sc.read_h5ad(dir / "adata_subset_COPD_MICA_meta.h5ad")
logger.info("AnnData object loaded: %s", adata)
logger.info("Data loaded successfully.")

logger.info("Check color palette defined in adata.uns for each level...")
level_list = ["level_1", "level_2", "level_3"]
palette_name = "tab20b"

for level in level_list:
    if level not in adata.obs.columns:
        logger.warning("'%s' not found in adata.obs.", level)
        continue

    palette_key = f"{level}_colors"

    # Ensure categorical
    adata.obs[level] = adata.obs[level].astype("category")
    categories = adata.obs[level].cat.categories.tolist()

    if palette_key not in adata.uns:
        logger.warning(
            "Color palette '%s' not found in adata.uns. Creating one.",
            palette_key,
        )

        colors = sns.color_palette(palette_name, n_colors=len(categories)).as_hex()
        adata.uns[palette_key] = colors

    else:
        existing_colors = adata.uns[palette_key]

        if len(existing_colors) != len(categories):
            logger.warning(
                "Palette '%s' length (%d) does not match number of categories (%d).",
                palette_key,
                len(existing_colors),
                len(categories),
            )

        logger.info("Color palette '%s' found in adata.uns.", palette_key)

# Check for ROI/sample imbalance across treatment arms
roi_summary = (
    adata.obs.groupby(["treatment_arm", "ROI"])
    .size()
    .groupby("treatment_arm")
    .agg(["count", "sum"])
)
logger.info("ROI/sample balance summary:\n%s", roi_summary)

# Remove clusters with less than 10 cells
logger.info("Filtering clusters with less than 10 cells per ROI...")
counts = (
    adata.obs.groupby(["ROI", "level_2"], observed=False)
    .size()
    .reset_index(name="count")
)

valid_pairs = counts[counts["count"] >= 10][["ROI", "level_2"]]

valid_pair_index = pd.MultiIndex.from_frame(valid_pairs[["ROI", "level_2"]])
adata_filtered = pd.MultiIndex.from_frame(adata.obs[["ROI", "level_2"]]).isin(
    valid_pair_index
)

adata = adata[adata_filtered].copy()

# Remove clusters with "unknown" annotation
logger.info("Removing clusters with 'unknown' annotation...")
adata = adata[~adata.obs["level_2"].str.contains("unknown", case=False)].copy()

# Remove clusters that are not in the sample type
logger.info("Removing clusters that are not in the sample type...")
cell_types_to_remove = [
    "AT1 cells",
    "AT2 cells",
    "Proliferating AT2 cells",
    "Alveolar fibroblasts",
    "Airway/Alveolar macrophages",
]
logger.info(f"Cell types to remove: {cell_types_to_remove}")
adata = adata[~adata.obs["level_2"].isin(cell_types_to_remove)].copy()
logger.info("Remaining cell types after filtering: %s", adata.obs["level_2"].unique())

# Subset to specific arm only for plotting
adata_treatment = adata[adata.obs["treatment_arm"].isin(["Treatment"])].copy()
adata_sham = adata[adata.obs["treatment_arm"].isin(["Sham"])].copy()

# Log number of cells
logger.info("Number of cells after filtering: %d", adata.n_obs)
logger.info("Number of genes after filtering: %d", adata.n_vars)
logger.info("Number of cells in Treatment arm: %d", adata_treatment.n_obs)
logger.info("Number of cells in Sham arm: %d", adata_sham.n_obs)

# Save filtered data
logger.info("Saving cleaned data...")
adata.write_h5ad(dir / "adata_subset_COPD_MICA_clean.h5ad")
adata_treatment.write_h5ad(dir / "adata_subset_COPD_MICA_Treatment.h5ad")
adata_sham.write_h5ad(dir / "adata_subset_COPD_MICA_Sham.h5ad")

logger.info("Data cleaning completed and saved successfully.")
