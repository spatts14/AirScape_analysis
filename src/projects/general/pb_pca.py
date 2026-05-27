"""Calculate PCA for pseudobulk data."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import sklearn

from utils.confidence_ellipse import confidence_ellipse
from utils.setup_logger import setup_logger

# Set up loggers
wd = "/rds/general/user/sep22/home/Projects/AirScape/HPC_jobs/general/"
logs_dir = Path(wd) / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir=logs_dir, log_name="pseudobulk")


# Set directory
dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/2026-03-27_analysis_run/"
)
data_dir = dir / "project_analysis/general/pb_data/"

# Set figure directory
folder_name = "pb_pca"
out_path = f"project_analysis/general/{folder_name}"

out_dir = dir / out_path
os.makedirs(out_dir, exist_ok=True)

# set fig dir for plots to save to
sc.settings.figdir = out_dir

# Set colors
cont_cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
cat_palette = sns.color_palette("Set2")

# Load data
logger.info("Loading data...")
df = pd.read_csv(data_dir / "pseudobulk_matrix_ROI.csv", index_col=0)
meta = pd.read_csv(data_dir / "pseudobulk_metadata_ROI.csv", index_col=0)
logger.info(f"Pseudobulk matrix shape: {df.shape}")
logger.info(f"Metadata shape: {meta.shape}")

# Format data
logger.info("Formatting data for PCA...")
X = df.T  # transpose to have samples as rows and genes as columns

# # Filer for lowly expressed genes to reduce noise in PCA
min_count = 10  # minimum count threshold across all samples
gene_sums = X.sum(axis=0)  # sum of counts for each gene across all samples
genes_to_keep = gene_sums[gene_sums > min_count].index
X = X[genes_to_keep]  # filter the matrix to keep only the selected genes
logger.info(
    f"Removed genes with less than {min_count} total counts across all samples."
)
logger.info(f"Number of filtered pseudobulk genes: {X.shape[1]}")

# Center and scale the data
logger.info("Log-transforming and scaling the data...")
X = np.log1p(X)  # log-transform the data
X_scaled = sklearn.preprocessing.StandardScaler().fit_transform(X)  # scale the data

# Calculate PCA
logger.info("Calculating PCA...")
pca = sklearn.decomposition.PCA(n_components=4)
pca_result = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2", "PC3", "PC4"], index=X.index)

# Add back metadata to PCA dataframe for plottings
pca_df = pca_df.join(meta)

# Plot PCA
col_list = meta.columns.tolist()  # get all columns from metadata

for col in col_list:
    if col not in pca_df.columns:
        logger.warning(f"Column '{col}' not found in pca_df. Skipping.")
        continue

    is_continuous = meta[col].dtype in ["int64", "float64"]
    color_arg = cont_cmap if is_continuous else cat_palette
    logger.info(
        f"Column '{col}' is {'continuous' if is_continuous else 'categorical'}."
    )

    # Plot PC1 vs PC2
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2", hue=col, s=50, palette=color_arg, ax=ax
    )

    # Confidence ellipses for categorical only
    if not is_continuous:
        groups = pca_df[col].unique()
        color_map = {g: cat_palette[i % len(cat_palette)] for i, g in enumerate(groups)}
        for group, color in color_map.items():
            subset = pca_df[pca_df[col] == group]
            if len(subset) >= 2:
                confidence_ellipse(
                    subset["PC1"].values,
                    subset["PC2"].values,
                    ax=ax,
                    n_std=2.0,
                    edgecolor=color,
                    linewidth=1.5,
                    linestyle="--",
                )

    ax.set_title(f"PCA of Pseudobulk Data: {col}")
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)", fontsize=14
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)", fontsize=14
    )
    ax.legend(title=col, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / f"{col}_pca.pdf")
    plt.close()
    logger.info(f"PCA plot saved to {col}_pca.pdf")

    # Plot pairplot of PC1-PC4
    sns.set_style("white")

    g = sns.pairplot(
        pca_df,
        vars=["PC1", "PC2", "PC3", "PC4"],
        corner=True,
        hue=col,
        palette=cat_palette,
        plot_kws={"s": 50},
    )

    # Move legend outside plot
    if g._legend is not None:
        g._legend.set_bbox_to_anchor((0.8, 0.9))
        g._legend.set_title(col)
        g._legend.set_frame_on(False)

    # Add title and adjust layout
    g.figure.suptitle(f"PCA Pairplot of Pseudobulk Data: {col}", y=1.02)
    g.figure.tight_layout()
    g.figure.subplots_adjust(right=0.85)  # must go after tight_layout
    g.figure.savefig(out_dir / f"{col}_pca_pairplot.pdf")
    plt.close()
    logger.info(f"PCA pairplot saved to {col}_pca_pairplot.pdf")


# PCA for condition and size variables
size_list = ["n_cells", "total_counts", "mean_transcripts"]
for size_col in size_list:
    if size_col not in meta.columns or "condition" not in meta.columns:
        continue

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="condition",
        size=size_col,
        sizes=(20, 400),
        palette=cat_palette,
        ax=ax,
    )
    ax.set_title(f"PCA of Pseudobulk Data: Condition and {size_col.capitalize()}")
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)", fontsize=14
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)", fontsize=14
    )
    ax.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / f"condition_{size_col}_pca.pdf")
    plt.close()

logger.info("PCA plots saved.")

logger.info("PCA analysis completed.")
