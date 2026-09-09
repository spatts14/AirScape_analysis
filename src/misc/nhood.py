"""Generate plots for neighborhood enrichment analysis results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set path
path = Path(
    "/Volumes/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/2026-03-27_analysis_run/spatial_stats/"
)

# Create output directory if it doesn't exist
outpath = path / "manual" / "nhood_manual"
outpath.mkdir(parents=True, exist_ok=True)

# Barplot path
barplot_path_zscore = outpath / "barplots_zscores"
barplot_path_zscore.mkdir(parents=True, exist_ok=True)

barplot_path_count = outpath / "barplots_counts"
barplot_path_count.mkdir(parents=True, exist_ok=True)

# Plot heatmap of z-scores for each condition
cmap_rnb = sns.color_palette("vlag", as_cmap=True)
condition_palettes = {
    "IPF": "#6A7FB5",  # slate blue
    "PM08": "#B07D4A",  # warm tan
    "COPD": "#7EB0B8",  # dusty teal
    "MICA": "#A67B8A",  # muted rose
}

# Import zscores for neighborhood enrichment analysis results
copd_df = pd.read_csv(
    path / "COPD_baseline_30" / "COPD_nhood_enrichment_zscores.csv",
    index_col=0,
)

mica_df = pd.read_csv(
    path / "MICA_30" / "MICA_nhood_enrichment_zscores.csv",
    index_col=0,
)

pm08_df = pd.read_csv(
    path / "PM08_30" / "PM08_nhood_enrichment_zscores.csv",
    index_col=0,
)

ipf_df = pd.read_csv(
    path / "IPF_30" / "IPF_nhood_enrichment_zscores.csv",
    index_col=0,
)

# Make a dictionary to store the dataframes for each condition
condition_dict = {"COPD": copd_df, "MICA": mica_df, "PM08": pm08_df, "IPF": ipf_df}

# Plot the z-scores for each condition
# Get list of all cell types across all conditions
cell_type_list = ipf_df.index.tolist()
celltype_dict = {}  # make empty dict to store combined dataframes for each cell type

# Combine dataframes for each cell type across conditions
for cell_type in cell_type_list:
    combined = pd.DataFrame(
        {
            condition: df.loc[cell_type]
            if cell_type in df.index
            else pd.Series(float("nan"), index=df.columns)
            for condition, df in condition_dict.items()
        }
    )

    celltype_dict[cell_type] = combined

for cell_type in cell_type_list:
    safe_cell_type = cell_type.replace("/", "_").replace(" ", "_")

    # Extract the dataframe for this cell type across conditions
    celltype_df = pd.DataFrame(celltype_dict[cell_type])

    # Transpose and melt the dataframe for plotting
    plot_df = (
        celltype_df.transpose().reset_index().rename(columns={"index": "Condition"})
    )
    plot_df = plot_df.melt(
        id_vars="Condition", var_name="Neighbor Cell Type", value_name="Z-score"
    )

    # Drop NaN values before plotting
    plot_df = plot_df.dropna()

    # Create the bar plot for this cell type
    sns.set_style("white")
    plt.figure(figsize=(20, 6))
    sns.barplot(
        data=plot_df,
        x="Neighbor Cell Type",
        y="Z-score",
        hue="Condition",
        palette=condition_palettes,
        hue_order=["MICA", "COPD", "PM08", "IPF"],
    )
    plt.title(f"{cell_type} - Neighborhood Enrichment Z-scores")
    plt.xticks(rotation=90)
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(barplot_path_zscore / f"{safe_cell_type}_nhood_enrichment_zscores.pdf")
    plt.close()


# Plot zscores for each condition as a clustermap
for condition, df in condition_dict.items():
    plot = sns.clustermap(
        df,
        row_cluster=False,
        col_cluster=False,
        cmap=cmap_rnb,
        figsize=(10, 10),
        cbar_pos=(0.02, 0.4, 0.03, 0.4),
        vmax=200,
        vmin=-200,
    )
    # plot.ax_heatmap.set_xlabel("Cell Type")
    # plot.ax_heatmap.set_ylabel("Cell Type")
    plot.ax_heatmap.set_title(f"{condition} - Neighborhood Enrichment Z-scores")
    plot.savefig(outpath / f"{condition}_nhood_enrichment_zscores_clustermap.pdf")
    plt.close()


# Plot count
# Import counts for neighborhood enrichment analysis results
copd_df = pd.read_csv(
    path / "COPD_baseline_30" / "COPD_nhood_enrichment_counts.csv",
    index_col=0,
)

mica_df = pd.read_csv(
    path / "MICA_30" / "MICA_nhood_enrichment_counts.csv",
    index_col=0,
)

pm08_df = pd.read_csv(
    path / "PM08_30" / "PM08_nhood_enrichment_counts.csv",
    index_col=0,
)

ipf_df = pd.read_csv(
    path / "IPF_30" / "IPF_nhood_enrichment_counts.csv",
    index_col=0,
)

# Make a dictionary to store the dataframes for each condition
condition_dict = {"COPD": copd_df, "MICA": mica_df, "PM08": pm08_df, "IPF": ipf_df}

# Plot the counts for each condition
# Get list of all cell types across all conditions
cell_type_list = ipf_df.index.tolist()
celltype_dict = {}  # make empty dict to store combined dataframes for each cell type

# Combine dataframes for each cell type across conditions
for cell_type in cell_type_list:
    combined = pd.DataFrame(
        {
            condition: df.loc[cell_type]
            if cell_type in df.index
            else pd.Series(float("nan"), index=df.columns)
            for condition, df in condition_dict.items()
        }
    )

    celltype_dict[cell_type] = combined

for cell_type in cell_type_list:
    safe_cell_type = cell_type.replace("/", "_").replace(" ", "_")

    # Extract the dataframe for this cell type across conditions
    celltype_df = pd.DataFrame(celltype_dict[cell_type])

    # Transpose and melt the dataframe for plotting
    plot_df = (
        celltype_df.transpose().reset_index().rename(columns={"index": "Condition"})
    )
    plot_df = plot_df.melt(
        id_vars="Condition", var_name="Neighbor Cell Type", value_name="Count"
    )

    # Drop NaN values before plotting
    plot_df = plot_df.dropna()

    # Create the bar plot for this cell type
    sns.set_style("white")
    plt.figure(figsize=(20, 6))
    sns.barplot(
        data=plot_df,
        x="Neighbor Cell Type",
        y="Count",
        hue="Condition",
        palette=condition_palettes,
        hue_order=["MICA", "COPD", "PM08", "IPF"],
    )
    plt.title(f"{cell_type} - Neighborhood Enrichment Counts")
    plt.xticks(rotation=90)
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.savefig(barplot_path_count / f"{safe_cell_type}_nhood_enrichment_counts.pdf")
    plt.close()
