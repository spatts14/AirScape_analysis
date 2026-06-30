"""Calculate PCA for pseudobulk data."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.airspace_colors import (
    condition_palette,
    diagnosis_palette,
)
from utils.safe_name import safe_name
from utils.setup_logger import setup_logger


def load_celltype_results(input_dir: Path):
    """Load saved pseudobulk matrices and metadata for each cell type."""
    results = []
    for cell_type_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        matrix_files = list(cell_type_dir.glob("*_pseudobulk_matrix_ROI.csv"))
        meta_files = list(cell_type_dir.glob("*_pseudobulk_metadata_ROI.csv"))

        if not matrix_files or not meta_files:
            continue

        matrix_file = matrix_files[0]
        meta_file = meta_files[0]
        cell_type_label = matrix_file.name.replace("_pseudobulk_matrix_ROI.csv", "")

        pb_sample = pd.read_csv(matrix_file, index_col=0)
        meta_df = pd.read_csv(meta_file, index_col=0)
        meta_df.index = meta_df.index.astype(str)

        common_samples = [
            sample for sample in pb_sample.columns if sample in meta_df.index
        ]
        if not common_samples:
            continue

        pb_sample = pb_sample[common_samples]
        meta_df = meta_df.loc[common_samples]

        results.append((cell_type_label, cell_type_dir, pb_sample, meta_df))
    return results


def plot_metric(df, x, y, cell_type, palette):
    """Plot a metric (y) by a grouping variable (x) for a given cell type."""
    plot_df = df[[x, y]].dropna().copy()
    plot_df[x] = plot_df[x].astype(str)

    fig, ax = plt.subplots(figsize=(6, 4))
    group_order = list(pd.unique(plot_df[x]))
    # Use the provided palette dict; fall back to a grey for unknown categories
    palette_to_use = [palette.get(g, "#888888") for g in group_order]

    sns.boxplot(
        data=plot_df,
        x=x,
        y=y,
        order=group_order,
        hue=x,
        palette=palette_to_use,
        saturation=0.25,
        ax=ax,
    )

    sns.stripplot(
        data=plot_df,
        x=x,
        y=y,
        hue=x,
        order=group_order,
        palette=palette_to_use,
        size=5,
        jitter=True,
        ax=ax,
        legend=False,
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{cell_type}: {y} by {x}")
    plt.tight_layout()

    return fig


def main():
    """Visualize number of cells per sample."""
    # Set directory
    path = Path(
        "/Volumes/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium"
    )

    # Set directories
    input_dir = path / "output" / "pb" / "pb_data_celltype"
    out_dir = path / "output" / "pb" / "num_cells_IPF_noCRD"
    out_dir.mkdir(parents=True, exist_ok=True)

    # set fig dir for plots to save to
    sc.settings.figdir = out_dir

    # Set up logger
    logs_dir = path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir=logs_dir, log_name="number_of_cells_per_sample")

    # Load data
    logger.info("Loading saved pseudobulk outputs...")
    results = load_celltype_results(input_dir)
    logger.info(f"Found {len(results)} cell type folders with saved outputs.")

    # Load total number of cells per sample
    total_cells = pd.read_csv(
        path / "output" / "pb" / "pb_data_concatenated" / "pseudobulk_metadata_ROI.csv",
        index_col=0,
    )

    # Metrics and Grouping columns
    y_metrics = ["n_cells", "normalized_num", "total_counts", "mean_transcripts"]
    group_cols = ["condition", "diagnosis"]
    group_palettes = {
        "condition": condition_palette,
        "diagnosis": diagnosis_palette,
    }

    # Remove COPD and MICAIII donors from the results list
    logger.info("Removing COPD and MICAIII donors from the results list...")
    results = [
        (
            cell_type,
            cell_type_dir,
            pb_sample[
                meta_df.index[~meta_df["diagnosis"].isin(["COPD", "HEALTHY", "NO_CRD"])]
            ],
            meta_df[~meta_df["diagnosis"].isin(["COPD", "HEALTHY", "NO_CRD"])],
        )
        for cell_type, cell_type_dir, pb_sample, meta_df in results
    ]

    # Add total number of cells and normalize to total cell number per sample
    logger.info(
        "Adding total number of cells and normalizing to total cell number per sample"
    )
    results = [
        (
            cell_type,
            cell_type_dir,
            pb_sample,
            (
                meta_df.assign(total_num_cells=total_cells["n_cells"]).assign(
                    normalized_num=lambda df: df["n_cells"] / df["total_num_cells"]
                )
            ),
        )
        for cell_type, cell_type_dir, pb_sample, meta_df in results
    ]

    logger.info("Plot absolute numbers of cells per sample.")
    for cell_type, cell_type_dir, pb_sample, meta_df in results:
        logger.info(f"Plotting cell type: {cell_type}...")

        # One subdirectory per cell type, named after its source directory
        cell_out_dir = out_dir / cell_type_dir.name
        cell_out_dir.mkdir(parents=True, exist_ok=True)

        for x in group_cols:
            if x not in meta_df.columns:
                logger.warning(f"{x} not in columns for {cell_type}. Skipping.")
                continue

            palette = group_palettes[x]  # dict mapping category → hex colour

            for y in y_metrics:
                if y not in meta_df.columns:
                    logger.warning(f"{y} not in columns for {cell_type}. Skipping.")
                    continue

                fig = plot_metric(
                    df=meta_df, x=x, y=y, cell_type=cell_type, palette=palette
                )
                fig.savefig(
                    cell_out_dir / f"{safe_name(cell_type)}_{y}_by_{x}.pdf",
                    bbox_inches="tight",
                )
                plt.close(fig)


if __name__ == "__main__":
    main()
