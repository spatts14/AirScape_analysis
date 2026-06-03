"""Generate plots for APT permutation results."""

import gc
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


# Define functions
def load_adjacency_p_values(input_dir: Path):
    """Load adjacency permutation test p-values for each ROI.

    Parameters:
        input_dir : Path
            Directory containing APT permutation test

    Returns:
        dict
            {roi_name: dataframe}

    """
    results = {}

    for file in sorted(input_dir.glob("adjacency_permutation_test_p_values_*.csv")):
        roi = file.stem.removeprefix("adjacency_permutation_test_p_values_")
        df = pd.read_csv(file, index_col=0)

        results[roi] = df

    return results


def make_clustermap(
    cell_type, celltype_dict, meta, meta_cols, cmap="vlag", fig_path=None
):
    """Make clustermap of APT scores for a given cell type.

    Args:
        cell_type (str): Cell type to plot
        celltype_dict (dict): Dictionary of dataframes for cell type across conditions
        meta (pd.DataFrame): Metadata dataframe with ROI information
        meta_cols (list): List of metadata columns to include in annotations
        cmap (str): Colormap for heatmap
        fig_path (Path): Path to save figure

    Returns:
        sns.ClusterGrid: ClusterGrid object containing the clustermap

    """
    safe_cell_type = cell_type.replace("/", "_").replace(" ", "_")

    # Build matrix
    celltype_df = pd.DataFrame(celltype_dict[cell_type])
    heatmap_df = celltype_df.apply(pd.to_numeric, errors="coerce").fillna(0)

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
        row_cluster=True,
        col_cluster=False,
        xticklabels=True,
        yticklabels=True,
    )

    g.figure.suptitle(f"{cell_type} - APT SES (p-val filtered)", y=1.02)

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
        fig_path / f"{safe_cell_type}_APT_SES_p_val_filtered.pdf",
        bbox_inches="tight",
    )
    plt.close()


# Base project path
base_path = Path(
    "/Volumes/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
)

# Input
input_dir = base_path / "output" / "muspan" / "adjacency_permutation_test_results"

# Output directories
outpath = base_path / "output" / "muspan" / "APT_plots"
heatmap_path = outpath / "heatmaps_zscores"
barplot_path_zscore = outpath / "barplots_zscores"

# Create directories
for path in [outpath, heatmap_path, barplot_path_zscore]:
    path.mkdir(parents=True, exist_ok=True)

# Plot heatmap of z-scores for each condition
cmap_rnb = sns.color_palette("vlag", as_cmap=True)
diagnosis_palette = {
    "IPF": "#6A7FB5",  # slate blue
    "LUNG_CANCER": "#B07D4A",  # warm tan
    "COPD": "#7EB0B8",  # dusty teal
    "HEALTHY": "#8EA882",  # sage
    "NO_CRD": "#A67B8A",  # muted rose
}

# Load metadata
meta = pd.read_csv(
    base_path / "data/meta/STx_meta_analysis_only_cleaned.csv", index_col=0
)

# Import adjacency permutation test
ROI_APT_dict = load_adjacency_p_values(input_dir)  #

# Get list of all cell types across conditions
cell_type_list = next(iter(ROI_APT_dict.values())).index.tolist()

# Make empty dict to store combined dataframes for each cell type
celltype_dict = {}

# Combine dataframes for each cell type across conditions
for cell_type in cell_type_list:
    combined = pd.DataFrame(
        {
            condition: df.loc[cell_type]
            if cell_type in df.index
            else pd.Series(float("nan"), index=df.columns)
            for condition, df in ROI_APT_dict.items()
        }
    )

    celltype_dict[cell_type] = combined

# Drop ROI names from celltype_dict
ROI_to_drop = ["MICA_III_319_315_311", "MICA_III_325_337_379"]
for cell_type in cell_type_list:
    celltype_dict[cell_type] = celltype_dict[cell_type].drop(columns=ROI_to_drop)


# PLOT HEATMAPS OF Z-SCORES FOR EACH CELL TYPE ACROSS CONDITIONS
for cell_type in cell_type_list:
    make_clustermap(
        cell_type=cell_type,
        celltype_dict=celltype_dict,
        meta=meta,
        meta_cols=["diagnosis"],
        fig_path=heatmap_path,
    )


# PLOT BARPLOTS OF Z-SCORES FOR EACH CELL TYPE ACROSS CONDITIONS

# List of metadata columns to merge with plot_df
metadata_cols = [
    "diagnosis",
]

for col in metadata_cols:
    if col not in meta.columns:
        raise ValueError(f"Metadata column '{col}' not found in meta dataframe")
    for cell_type in cell_type_list:
        safe_cell_type = cell_type.replace("/", "_").replace(" ", "_")

        # make directory for this cell type if it doesn't exist
        celltype_dir = barplot_path_zscore / safe_cell_type
        celltype_dir.mkdir(parents=True, exist_ok=True)

        # Extract the dataframe for this cell type across conditions
        celltype_df = pd.DataFrame(celltype_dict[cell_type])

        # Transpose and melt the dataframe for plotting
        plot_df = celltype_df.transpose().reset_index().rename(columns={"index": "ROI"})

        plot_df = plot_df.melt(
            id_vars="ROI",
            var_name="Neighbor Cell Type",
            value_name="SES_p_val_filtered",
        )

        # # Add condition column by extracting from ROI name
        plot_df = plot_df.merge(meta, left_on="ROI", right_index=True)

        # Drop NaN values before plotting
        plot_df = plot_df.dropna(subset=["SES_p_val_filtered"])

        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(35, 6))

        sns.stripplot(
            data=plot_df,
            x="Neighbor Cell Type",
            y="SES_p_val_filtered",
            hue=col,
            dodge=True,
            alpha=0.5,
            ax=ax,
        )

        sns.boxenplot(
            data=plot_df,
            x="Neighbor Cell Type",
            y="SES_p_val_filtered",
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

        ax.set_title(f"{cell_type} - ADT SES (p-val filtered)")
        ax.set_ylabel("SES (p-val filtered)")
        ax.tick_params(axis="x", rotation=90)

        plt.tight_layout()
        plt.savefig(
            celltype_dir / f"{safe_cell_type}_APT_SES_p_val_filtered.pdf",
            bbox_inches="tight",
        )


# Release memory
del ROI_APT_dict
del celltype_dict
del meta

gc.collect()
