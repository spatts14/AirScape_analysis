"""Add manual metadata to adata.obs to annotated adata."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import scanpy as sc

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.airspace_colors import level_2_palette


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


# from utils.seed_everything import seed_everything
# from utils.setup_logger import setup_logger

# Set directory
path = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium"
)
dir = path / "output/AIRSCAPE/"

logs_dir = Path(path) / "logs"
logger = setup_logger(log_dir=logs_dir, log_name="metadata_addition")

# Set figure directory
folder_name = "adata_final_object"
out_dir = dir / folder_name
os.makedirs(out_dir, exist_ok=True)

# Load data
logger.info("Loading data...")
adata = sc.read_h5ad(dir / "annotate/adata_level_2_level_3.h5ad")  # full dataset

logger.info(f"adata shape: {adata.shape}")
logger.info(f"adata.obs columns: {adata.obs.columns.tolist()}")
logger.info(adata)

# Total number of columns in adata.obs after addition
logger.info(
    f"Total number of columns in adata.obs before addition: {adata.obs.shape[1]}"
)

# Load metadata
logger.info("Loading metadata...")
meta_dir = dir / "data/meta"
metadata = pd.read_csv(
    path / "data/meta/STx_meta_analysis_only_cleaned.csv", index_col="ROI"
)

# Check which metadata columns are already in adata.obs
logger.info("Checking which metadata columns are already in adata.obs...")
for col in metadata.columns:
    if col in adata.obs.columns:
        logger.warning(f"Column '{col}' already exists in adata.obs. Skipping.")
    else:
        logger.info(f"Column '{col}' does not exist in adata.obs. Will be added.")

logger.info("Adding metadata to adata.obs...")
# Make a list of columns to add
cols_to_add = metadata.columns.difference(adata.obs.columns)

# Add metadata columns to adata.obs
adata.obs = adata.obs.join(metadata[cols_to_add], on="ROI", how="left")


# confirm that metadata has been added successfully
logger.info("Confirming that metadata has been added successfully...")
for col in metadata.columns:
    if col in adata.obs.columns:
        logger.info(f"Column '{col}' successfully added to adata.obs.")
    else:
        logger.error(f"Column '{col}' not found in adata.obs after addition.")
# Total number of columns in adata.obs after addition
logger.info(
    f"Total number of columns in adata.obs after addition: {adata.obs.shape[1]}"
)

# Drop column named "ID" if it exists in adata.obs
if "ID" in adata.obs.columns:
    adata.obs.drop(columns=["ID"], inplace=True)
    logger.info("Column 'ID' found and dropped from adata.obs.")
else:
    logger.info("Column 'ID' not found in adata.obs. No need to drop.")

# Clean column names first
adata.obs.columns = adata.obs.columns.str.strip()

# Make a list of all the unique values in the "treatment_arm" column of adata.obs
if "treatment_arm" in adata.obs.columns:
    unique_treatment_arms = adata.obs["treatment_arm"].unique()
    logger.info(f"Unique values in 'treatment_arm' column: {unique_treatment_arms}")


# Add color palette for treatment arms to adata.uns
adata.uns["level_2_colors"] = [
    level_2_palette[ct] for ct in adata.obs["level_2"].cat.categories
]
logger.info(
    "Color palette for 'level_2' annotation added to adata.uns['level_2_colors']."
)

# Confirm that the color palette has been added successfully
if "level_2_colors" in adata.uns:
    logger.info(
        "Color palette for 'level_2' annotation successfully added to adata.uns."
    )
else:
    logger.error(
        "Color palette for 'level_2' annotation not found in adata.uns after addition."
    )

# Confirm level_2 still exists in adata.obs and has the same categories
if "level_2" in adata.obs.columns:
    logger.info("'level_2' annotation still exists in adata.obs after addition.")
    logger.info(f"Categories in 'level_2': {adata.obs['level_2'].cat.categories}")
else:
    logger.error("'level_2' annotation not found in adata.obs after addition.")

# Save updated adata object as zarr file
output_file = out_dir / "adata_with_metadata.h5ad"
adata.write_h5ad(output_file)
logger.info(f"Updated adata object saved to {output_file}")

# save as .zarr file as well
output_zarr_file = out_dir / "adata_with_metadata.zarr"
adata.write_zarr(output_zarr_file)
logger.info(f"Updated adata object also saved to {output_zarr_file}")
