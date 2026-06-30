"""Calculate PCA for pseudobulk data."""

import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import mannwhitneyu

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


def run_pairwise_mannwhitney(df, x, y, alpha_level=0.05):
    """Run Mann-Whitney U test for every pair of groups in column x, on values in column y.

    Returns a list of (group_1, group_2, stat, p_value) tuples, restricted to
    pairs with p_value < alpha_level if you want to filter later; here we
    return all comparisons and let the caller decide what to annotate.
    """
    plot_df = df[[x, y]].dropna().copy()
    plot_df[x] = plot_df[x].astype(str)

    groups = list(pd.unique(plot_df[x]))
    test_results = []

    for group_1, group_2 in combinations(groups, 2):
        group_1_data = plot_df.loc[plot_df[x] == group_1, y].to_numpy().astype(float)
        group_2_data = plot_df.loc[plot_df[x] == group_2, y].to_numpy().astype(float)

        group_1_data = group_1_data[~np.isnan(group_1_data)]
        group_2_data = group_2_data[~np.isnan(group_2_data)]

        if len(group_1_data) < 1 or len(group_2_data) < 1:
            continue

        stat, p_value = mannwhitneyu(group_1_data, group_2_data)
        test_results.append((group_1, group_2, stat, p_value))

    return test_results


def plot_metric(df, x, y, cell_type, palette, alpha_level=0.05):
    """Plot a metric (y) by a grouping variable (x) for a given cell type,
    with Mann-Whitney U test results shown in the title and significance
    brackets drawn between groups.
    """
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

    # Run pairwise Mann-Whitney U tests
    test_results = run_pairwise_mannwhitney(df, x, y, alpha_level=alpha_level)

    # Build a title that includes the test stat/p-value for each pair
    title_lines = [f"{cell_type}: {y} by {x}"]
    for group_1, group_2, stat, p_value in test_results:
        title_lines.append(f"{group_1} vs {group_2}: p={p_value:.3g}")
    ax.set_title("\n".join(title_lines), fontsize=9)

    # --- Make room above the data for brackets/asterisks ---
    n_sig = sum(p_value < alpha_level for *_, p_value in test_results)
    if n_sig > 0:
        y_min, y_max = ax.get_ylim()
        data_range = y_max - y_min
        # Reserve ~15% of the current range per significant bracket,
        # plus a base 10% buffer above the highest data point.
        headroom = data_range * (0.15 + 0.15 * n_sig)
        ax.set_ylim(y_min, y_max + headroom)

    # Add significance brackets between groups, stacked if more than one pair
    n_groups = len(group_order)
    base_y = 0.95
    step = 0.08
    sig_idx = 0
    for group_1, group_2, stat, p_value in test_results:
        if p_value < alpha_level:
            x1 = group_order.index(group_1)
            x2 = group_order.index(group_2)

            xy_left = ((x1 + 0.5) / n_groups, base_y - sig_idx * step)
            xy_right = ((x2 + 0.5) / n_groups, base_y - sig_idx * step)

            ax.annotate(
                "",
                xy=xy_left,
                xycoords="axes fraction",
                xytext=xy_right,
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color="k", lw=1),
            )
            ax.text(
                (xy_left[0] + xy_right[0]) / 2,
                base_y - sig_idx * step - 0.001,
                "*",
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                color="k",
            )
            sig_idx += 1

    # Leave extra room at the top for the multi-line title too
    fig.subplots_adjust(top=0.80 if len(title_lines) > 1 else 0.88)

    return fig, test_results


def main():
    """Visualize number of cells per sample."""
    # Set directory
    path = Path(
        "/Volumes/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium"
    )

    # Set directories
    input_dir = path / "output" / "pb" / "pb_data_celltype"
    out_dir = path / "output" / "pb" / "num_cells_IPF_noCRD_TEST"
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
    alpha_level = 0.05

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
    stats_records = []
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

                fig, test_results = plot_metric(
                    df=meta_df,
                    x=x,
                    y=y,
                    cell_type=cell_type,
                    palette=palette,
                    alpha_level=alpha_level,
                )
                fig.savefig(
                    cell_out_dir / f"{safe_name(cell_type)}_{y}_by_{x}.pdf",
                    bbox_inches="tight",
                )
                plt.close(fig)

                for group_1, group_2, stat, p_value in test_results:
                    stats_records.append(
                        {
                            "cell_type": cell_type,
                            "group_col": x,
                            "metric": y,
                            "group_1": group_1,
                            "group_2": group_2,
                            "statistic": stat,
                            "p_value": p_value,
                            "significant": p_value < alpha_level,
                        }
                    )

    # Save all Mann-Whitney results to a single CSV for review
    stats_df = pd.DataFrame(stats_records)
    stats_df.to_csv(out_dir / "mannwhitney_results.csv", index=False)
    logger.info(
        f"Saved Mann-Whitney U test results to {out_dir / 'mannwhitney_results.csv'}"
    )


if __name__ == "__main__":
    main()
