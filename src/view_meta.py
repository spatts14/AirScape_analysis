"""Export ROIs and metadata from spatial transcriptomics data to a spreadsheet."""

import logging
from datetime import datetime
from pathlib import Path

import anndata as ad


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


# Set directory
path = Path(
    "/Volumes/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/r"
)
output_dir = path / "output/AIRSCAPE/"
output_dir.mkdir(parents=True, exist_ok=True)

logs_dir = path / "logs"
logger = setup_logger(log_dir=logs_dir, log_name="metadata_print")


# Load data
logger.info("Loading data...")
adata = ad.read_zarr(
    output_dir / "adata_final_object/adata_with_metadata.zarr"
)  # full dataset

logger.info(f"adata shape: {adata.shape}")
logger.info(f"adata.obs columns: {adata.obs.columns.tolist()}")
logger.info(adata)


# ---- CONFIG: adjust these to match your adata.obs columns ----
ROI_COL = "ROI"  # column in adata.obs that identifies the ROI
META_COLS = (
    None  # list of ROI-level metadata columns to include, or None to auto-detect
)

# ---- Build ROI-level metadata table ----

obs = adata.obs.copy()

if ROI_COL not in obs.columns:
    raise ValueError(
        f"Column '{ROI_COL}' not found in adata.obs. "
        f"Available columns: {list(obs.columns)}"
    )

candidate_cols = [c for c in obs.columns if c != ROI_COL]

# Identify which columns are actually constant within each ROI (i.e. true
# ROI-level metadata) versus columns that vary at the cell level (e.g. cell
# type, spatial coordinates, QC metrics). Only the former belong in a
# one-row-per-ROI table.
nunique_per_roi = obs.groupby(ROI_COL, observed=True)[candidate_cols].nunique()
roi_level_cols = [c for c in candidate_cols if nunique_per_roi[c].max() == 1]
cell_level_cols = [c for c in candidate_cols if c not in roi_level_cols]

if cell_level_cols:
    logger.warning(
        f"Excluding these columns because they vary within at least one ROI "
        f"(i.e. they are cell-level, not ROI-level): {cell_level_cols}"
    )

if META_COLS is None:
    META_COLS = roi_level_cols
else:
    # If the user explicitly requested columns, warn (but still honor it)
    # for any that aren't actually constant within an ROI.
    requested_cell_level = [c for c in META_COLS if c in cell_level_cols]
    if requested_cell_level:
        logger.warning(
            f"Requested META_COLS include cell-level columns; values shown "
            f"will be the first cell's value per ROI, not a true ROI-level "
            f"summary: {requested_cell_level}"
        )

# One row per ROI, with only true ROI-level metadata columns
roi_metadata = obs.groupby(ROI_COL, observed=True)[META_COLS].first().reset_index()

# Add a count of cells/spots per ROI, useful for QC
roi_counts = obs.groupby(ROI_COL, observed=True).size().reset_index(name="n_cells")
roi_metadata = roi_metadata.merge(roi_counts, on=ROI_COL)

# ---- Write to Excel ----

output_filename = "roi_metadata.xlsx"
output_path = output_dir / output_filename
roi_metadata.to_excel(output_path, index=False, sheet_name="ROI Metadata")

logger.info(f"Wrote {len(roi_metadata)} ROIs to {output_path}")
print(roi_metadata.head())
