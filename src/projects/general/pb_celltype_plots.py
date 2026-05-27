"""Plot metrics and PCA for pseudobulk cell type data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

from utils.confidence_ellipse import confidence_ellipse
from utils.safe_name import safe_name
from utils.seed_everything import seed_everything
from utils.setup_logger import setup_logger


def plot_metric(df, x, y, cell_type, palette_name="Set2"):
    """Plot a metric (y) by a grouping variable (x) for a given cell type."""
    plot_df = df[[x, y]].dropna().copy()
    plot_df[x] = plot_df[x].astype(str)

    fig, ax = plt.subplots(figsize=(6, 4))
    group_order = list(pd.unique(plot_df[x]))
    palette_to_use = sns.color_palette(palette_name, n_colors=len(group_order))

    sns.boxenplot(
        data=plot_df,
        x=x,
        y=y,
        order=group_order,
        palette=palette_to_use,
        saturation=0.25,
        ax=ax,
    )

    sns.stripplot(
        data=plot_df,
        x=x,
        y=y,
        order=group_order,
        color="0.2",
        size=2.5,
        alpha=0.7,
        jitter=True,
        ax=ax,
        legend=False,
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{cell_type}: {y} by {x}")
    plt.tight_layout()

    return fig


def load_celltype_results(out_dir: Path):
    """Load saved pseudobulk matrices and metadata for each cell type."""
    results = []
    for cell_type_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
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


def main():
    """Plot metrics and PCA from saved pseudobulk outputs."""
    seed_everything(19960915)

    path = Path(
        "/Volumes/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
    )

    logs_dir = Path(path) / "logs"
    logger = setup_logger(log_dir=logs_dir, log_name="pseudobulk_plots")

    dir = path / "output/2026-03-27_analysis_run"
    out_dir = dir / "project_analysis/general/pb_data_celltype"

    cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
    cat_palette = sns.color_palette("Set2")

    y_metrics = ["n_cells", "total_counts", "mean_transcripts"]
    group_cols = ["condition", "timepoint_label", "treatment_arm"]
    col_list = [
        "sample_ID",
        "condition",
        "timepoint_label",
        "batch",
        "treatment_arm",
        "lung_location",
        "biopsy_type",
        "diagnosis",
        "sex",
        "age",
    ]

    logger.info("Loading saved pseudobulk outputs...")
    results = load_celltype_results(out_dir)
    logger.info(f"Found {len(results)} cell type folders with saved outputs.")

    for cell_type, cell_type_dir, pb_sample, meta_df in results:
        logger.info(f"Plotting cell type: {cell_type}...")

        for x in group_cols:
            if x not in meta_df.columns:
                logger.warning(f"{x} not in columns for {cell_type}. Skipping.")
                continue

            for y in y_metrics:
                if y not in meta_df.columns:
                    logger.warning(f"{y} not in columns for {cell_type}. Skipping.")
                    continue

                fig = plot_metric(df=meta_df, x=x, y=y, cell_type=cell_type)
                fig.savefig(
                    cell_type_dir / f"{safe_name(cell_type)}_{y}_by_{x}.pdf",
                    bbox_inches="tight",
                )
                plt.close(fig)

        logger.info("Formatting data for PCA...")
        X = pb_sample.T

        min_count = 10
        gene_sums = X.sum(axis=0)
        genes_to_keep = gene_sums[gene_sums > min_count].index
        X = X[genes_to_keep]
        logger.info(
            f"Removed genes with less than {min_count} total counts across all samples."
        )
        logger.info(f"Number of filtered pseudobulk genes: {X.shape[1]}")

        logger.info("Log-transforming and scaling the data...")
        X = np.log1p(X)
        X_scaled = sklearn.preprocessing.StandardScaler().fit_transform(X)

        logger.info("Calculating PCA...")
        pca = sklearn.decomposition.PCA(n_components=4)
        pca_result = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(
            pca_result, columns=["PC1", "PC2", "PC3", "PC4"], index=X.index
        )
        pca_df = pca_df.join(meta_df)

        for col in col_list:
            if col not in meta_df.columns:
                logger.warning(f"Column '{col}' not found in metadata. Skipping plot.")
                continue

            logger.info(f"Plotting PCA colored by '{col}'...")
            sns.set_style("white")
            fig, ax = plt.subplots(figsize=(6, 5))

            is_continuous = pd.api.types.is_numeric_dtype(pca_df[col])
            if is_continuous:
                scatter = ax.scatter(
                    pca_df["PC1"],
                    pca_df["PC2"],
                    c=pca_df[col],
                    cmap=cmap,
                    s=50,
                )
                fig.colorbar(scatter, ax=ax, label=col)
            else:
                groups = pca_df[col].dropna().unique()
                color_map = {
                    g: cat_palette[i % len(cat_palette)] for i, g in enumerate(groups)
                }
                sns.scatterplot(
                    data=pca_df,
                    x="PC1",
                    y="PC2",
                    hue=col,
                    s=50,
                    palette=cat_palette,
                    ax=ax,
                )

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

                ax.legend(title=col, bbox_to_anchor=(1.05, 1), loc="upper left")

            ax.set_title(f"PCA of Pseudobulk Data: {col}")
            ax.set_xlabel(
                f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)",
                fontsize=14,
            )
            ax.set_ylabel(
                f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)",
                fontsize=14,
            )
            plt.tight_layout()
            plt.savefig(cell_type_dir / f"{col}_{safe_name(cell_type)}_pca.pdf")
            plt.close(fig)
            logger.info(f"PCA plot saved to {col}_{safe_name(cell_type)}_pca.pdf")

        for col in col_list:
            if col not in meta_df.columns:
                continue

            logger.info(f"Plotting pairplot colored by '{col}'...")
            sns.set_style("white")
            g = sns.pairplot(
                pca_df,
                vars=["PC1", "PC2", "PC3", "PC4"],
                hue=col,
                palette=cat_palette,
                plot_kws={"s": 50},
                corner=True,
            )

            if g._legend is not None:
                g._legend.set_bbox_to_anchor((0.8, 0.9))
                g._legend.set_title(col)
                g._legend.set_frame_on(False)

            g.figure.suptitle(f"PCA Pairplot of Pseudobulk Data: {col}", y=1.02)
            g.figure.tight_layout()
            g.figure.subplots_adjust(right=0.85)
            g.figure.savefig(
                cell_type_dir / f"{col}_{safe_name(cell_type)}_pca_pairplot.pdf"
            )
            plt.close(g.figure)
            logger.info(
                f"PCA pairplot saved to {col}_{safe_name(cell_type)}_pca_pairplot.pdf"
            )

        logger.info("PCA plots saved.")


if __name__ == "__main__":
    main()
