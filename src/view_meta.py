"""Export ROIs and metadata from a spatial transcriptomics to an spreadsheet."""

import logging
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad

sys.path.append(str(Path(__file__).resolve().parents[1]))


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
dir = path / "output/AIRSCAPE/"

logs_dir = Path(path) / "logs"
logger = setup_logger(log_dir=logs_dir, log_name="metadata_print")


# Load data
logger.info("Loading data...")
adata = ad.read_zarr(
    dir / "adata_final_object/adata_with_metadata.zarr"
)  # full dataset

logger.info(f"adata shape: {adata.shape}")
logger.info(f"adata.obs columns: {adata.obs.columns.tolist()}")
logger.info(adata)


# ---- CONFIG: adjust these to match your adata.obs columns ----
ROI_COL = "ROI"  # column in adata.obs that identifies the ROI
META_COLS = None  # list of metadata columns to include, or None for all columns

# ---- Build ROI-level metadata table ----

obs = adata.obs.copy()

if ROI_COL not in obs.columns:
    raise ValueError(
        f"Column '{ROI_COL}' not found in adata.obs. "
        f"Available columns: {list(obs.columns)}"
    )

if META_COLS is None:
    META_COLS = [c for c in obs.columns if c != ROI_COL]

# Group by ROI, taking the first value of each metadata column
# (assumes metadata is constant within each ROI; if not, this will
# just show the first occurrence per ROI)
roi_metadata = obs.groupby(ROI_COL, observed=True)[META_COLS].first().reset_index()

# Add a count of cells/spots per ROI, useful for QC
roi_counts = obs.groupby(ROI_COL, observed=True).size().reset_index(name="n_cells")
roi_metadata = roi_metadata.merge(roi_counts, on=ROI_COL)

# ---- Write to Excel ----

output_path = "roi_metadata.xlsx"
roi_metadata.to_excel(dir / output_path, index=False, sheet_name="ROI Metadata")

print(f"Wrote {len(roi_metadata)} ROIs to {output_path}")
print(roi_metadata.head())
