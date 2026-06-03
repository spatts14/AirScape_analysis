"""Generate plots for neighborhood enrichment analysis results."""

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


# Define functions
def load_nhood_enrichment_counts(input_dir: Path):
    """Load neighborhood enrichment counts for each ROI.

    Parameters:
        input_dir : Path
            Directory containing ROI subdirectories with count files

    Returns:
        dict
            {roi_name: dataframe}

    """
    results = {}

    # iterate over ROI subdirectories
    for roi_dir in input_dir.iterdir():
        if not roi_dir.is_dir():
            continue

        roi_name = roi_dir.name

        # wildcard search for count file
        count_files = list(roi_dir.glob("*_nhood_enrichment_counts.csv"))

        if not count_files:
            continue

        if len(count_files) > 1:
            raise ValueError(
                f"Multiple count files found for {roi_name}: {count_files}"
            )

        count_file = count_files[0]

        # read file
        df = pd.read_csv(count_file, index_col=0)

        results[roi_name] = df

    return results


def make_clustermap(
    cell_type,
    celltype_dict,
    meta,
    meta_cols,
    log_transform=False,
    cmap="vlag",
    fig_path=None,
):
    """Make clustermap of neighborhood enrichment counts for a given cell type.

    Args:
        cell_type (str): Cell type to plot
        celltype_dict (dict): Dictionary of dataframes for cell type across conditions
        meta (pd.DataFrame): Metadata dataframe with ROI information
        meta_cols (list): List of metadata columns to include in annotations
        cmap (str): Colormap for heatmap
        fig_path (Path): Path to save figure
        log_transform (bool): Whether to log-transform counts for better visualization

    Returns:
        sns.ClusterGrid: ClusterGrid object containing the clustermap

    """
    safe_cell_type = cell_type.replace("/", "_").replace(" ", "_")

    # Build matrix
    celltype_df = pd.DataFrame(celltype_dict[cell_type])
    heatmap_df = celltype_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # log transform counts for better visualization (add 1 to avoid log(0))
    if log_transform:
        heatmap_df = heatmap_df.map(lambda x: np.log1p(x))

    # Validate metadata
    for col in meta_cols:
        if col not in meta.columns:
            raise ValueError(f"Missing metadata column: {col}")

    meta_subset = meta.loc[heatmap_df.columns, meta_cols]

    # Build color annotations + store palettes
    col_colors = pd.DataFrame(index=heatmap_df.columns)
    palettes = {}

    for col in meta_cols:
        # Unique values
        unique_vals = meta_subset[col].unique()

        # Number of unique values
        length = len(unique_vals)

        palette = dict(zip(unique_vals, sns.color_palette("husl", length)))

        palettes[col] = palette
        col_colors[col] = meta_subset[col].map(palette)

    # Plot
    g = sns.clustermap(
        heatmap_df,
        cmap=cmap,
        center=0,
        linewidths=0.5,
        figsize=(12, 8),
        col_colors=col_colors,
        xticklabels=True,
        yticklabels=True,
    )

    g.figure.suptitle(f"{cell_type} - Neighborhood Enrichment Counts", y=1.02)

    # Build legend (outside plot)
    legend_handles = []

    for col in meta_cols:
        for level, color in palettes[col].items():
            legend_handles.append(Patch(facecolor=color, label=f"{col}: {level}"))

    g.ax_heatmap.legend(
        handles=legend_handles,
        title="Metadata",
        bbox_to_anchor=(1.6, 1),
        loc="upper left",
        frameon=False,
    )

    # Save figure
    plt.savefig(
        fig_path / f"{safe_cell_type}_nhood_enrichment_clustermap_counts.pdf",
        bbox_inches="tight",
    )
    plt.close()


# Base project path
base_path = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium"
)

# Analysis paths
analysis_path = base_path / "output" / "2026-03-27_analysis_run"
spatial_stats_path = analysis_path / "spatial_stats"

# Input
input_dir = spatial_stats_path / "ROI_spatial_statistics" / "30_radius"

# Output directories
outpath = (
    spatial_stats_path / "ROI_spatial_statistics" / "30_radius_manual" / "nhood_manual"
)
heatmap_path = outpath / "heatmaps_counts"
barplot_path_count = outpath / "barplots_counts"

# Create directories
for path in [outpath, heatmap_path, barplot_path_count]:
    path.mkdir(parents=True, exist_ok=True)

# Plot heatmap of z-scores for each condition
cmap_rnb = sns.color_palette("vlag", as_cmap=True)
cmap = sns.color_palette("Blues", as_cmap=True)
condition_palettes = {
    "IPF": "#6A7FB5",  # slate blue
    "PM08": "#B07D4A",  # warm tan
    "COPD": "#7EB0B8",  # dusty teal
    "MICA": "#A67B8A",  # muted rose
}

# Load metadata
meta = pd.read_csv(
    base_path / "data/meta/STx_meta_analysis_only_cleaned.csv", index_col=0
)

# Import counts for neighborhood enrichment analysis results
ROI_count_dict = load_nhood_enrichment_counts(input_dir)

# Get list of all cell types across conditions
cell_type_list = next(iter(ROI_count_dict.values())).index.tolist()

# Make empty dict to store combined dataframes for each cell type
celltype_dict = {}

# Combine dataframes for each cell type across conditions
for cell_type in cell_type_list:
    combined = pd.DataFrame(
        {
            condition: df.loc[cell_type]
            if cell_type in df.index
            else pd.Series(float("nan"), index=df.columns)
            for condition, df in ROI_count_dict.items()
        }
    )

    celltype_dict[cell_type] = combined


# PLOT HEATMAPS OF COUNTS FOR EACH CELL TYPE ACROSS CONDITIONS
for cell_type in cell_type_list:
    make_clustermap(
        cell_type=cell_type,
        celltype_dict=celltype_dict,
        meta=meta,
        cmap=cmap,
        log_transform=False,
        meta_cols=["diagnosis", "time_point", "treatment_arm"],
        fig_path=heatmap_path,
    )


# PLOT BARPLOTS OF COUNTS FOR EACH CELL TYPE ACROSS CONDITIONS

# Timepoint and treatment arm analysis
col = "treatment_arm_time_point"
order_tp = [
    "SHAM_V1",
    "SHAM_V2",
    "SHAM_V3",
    "TREATMENT_V1",
    "TREATMENT_V2",
    "TREATMENT_V3",
]
for cell_type in cell_type_list:
    safe_cell_type = cell_type.replace("/", "_").replace(" ", "_")

    # make directory for this cell type if it doesn't exist
    celltype_dir = barplot_path_count / safe_cell_type
    celltype_dir.mkdir(parents=True, exist_ok=True)

    # Extract the dataframe for this cell type across conditions
    celltype_df = pd.DataFrame(celltype_dict[cell_type])

    # Transpose and melt the dataframe for plotting
    plot_df = celltype_df.transpose().reset_index().rename(columns={"index": "ROI"})

    plot_df = plot_df.melt(
        id_vars="ROI", var_name="Neighbor Cell Type", value_name="Count"
    )

    # # Add condition column by extracting from ROI name
    plot_df = plot_df.merge(meta, left_on="ROI", right_index=True)

    # Drop NaN values before plotting
    plot_df = plot_df.dropna(subset=["Count"])

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(35, 6))

    sns.stripplot(
        data=plot_df,
        x="Neighbor Cell Type",
        y="Count",
        hue=col,
        dodge=True,
        alpha=0.5,
        hue_order=order_tp,
        ax=ax,
    )

    sns.boxenplot(
        data=plot_df,
        x="Neighbor Cell Type",
        y="Count",
        hue=col,
        dodge=True,
        hue_order=order_tp,
        ax=ax,
    )

    # Remove duplicate legends
    handles, labels = ax.get_legend_handles_labels()
    n = len(plot_df[col].unique())

    ax.legend(
        handles[:n],
        labels[:n],
        title=col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    ax.set_title(f"{cell_type} - Neighborhood Enrichment Counts")
    ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.savefig(celltype_dir / f"{safe_cell_type}_{col}_nhood_enrichment_counts.pdf")
    plt.close()


# List of metadata columns to merge with plot_df
metadata_cols = [
    "diagnosis",
    "lung_location",
    "time_point",
    "treatment_arm",
    "biopsy_type",
]

for col in metadata_cols:
    if col not in meta.columns:
        raise ValueError(f"Metadata column '{col}' not found in meta dataframe")
    for cell_type in cell_type_list:
        safe_cell_type = cell_type.replace("/", "_").replace(" ", "_")

        # make directory for this cell type if it doesn't exist
        celltype_dir = barplot_path_count / safe_cell_type
        celltype_dir.mkdir(parents=True, exist_ok=True)

        # Extract the dataframe for this cell type across conditions
        celltype_df = pd.DataFrame(celltype_dict[cell_type])

        # Transpose and melt the dataframe for plotting
        plot_df = celltype_df.transpose().reset_index().rename(columns={"index": "ROI"})

        plot_df = plot_df.melt(
            id_vars="ROI", var_name="Neighbor Cell Type", value_name="Count"
        )

        # # Add condition column by extracting from ROI name
        plot_df = plot_df.merge(meta, left_on="ROI", right_index=True)

        # Drop NaN values before plotting
        plot_df = plot_df.dropna(subset=["Count"])

        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(35, 6))

        sns.stripplot(
            data=plot_df,
            x="Neighbor Cell Type",
            y="Count",
            hue=col,
            dodge=True,
            alpha=0.5,
            ax=ax,
        )

        sns.boxenplot(
            data=plot_df,
            x="Neighbor Cell Type",
            y="Count",
            hue=col,
            dodge=True,
            ax=ax,
        )

        # Remove duplicate legends
        handles, labels = ax.get_legend_handles_labels()
        n = len(plot_df[col].unique())

        ax.legend(
            handles[:n],
            labels[:n],
            title=col,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )

        ax.set_title(f"{cell_type} - Neighborhood Enrichment Counts")
        ax.tick_params(axis="x", rotation=90)

        plt.tight_layout()
        plt.savefig(
            celltype_dir / f"{safe_cell_type}_{col}_nhood_enrichment_counts.pdf"
        )
        plt.close()

# Release memory
del ROI_count_dict
del celltype_dict
del meta

gc.collect()
