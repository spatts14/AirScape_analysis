"""Plot metrics and PCA for pseudobulk cell type data."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

sys.path.append(str(Path(__file__).resolve().parents[2]))

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
    palette_map = dict(zip(group_order, palette_to_use))

    sns.boxplot(
        data=plot_df,
        x=x,
        y=y,
        order=group_order,
        hue=x,
        dodge=False,
        palette=palette_map,
        legend=False,
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
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{cell_type}: {y} by {x}")
    plt.tight_layout()

    return fig


def load_celltype_results(input_dir: Path):
    """Load saved pseudobulk matrices and metadata for each cell type."""
    results = []

    for cell_type_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        matrix_files = list(cell_type_dir.glob("*_pseudobulk_matrix_ROI.csv"))
        meta_files = list(cell_type_dir.glob("*_pseudobulk_metadata_ROI.csv"))

        if not matrix_files or not meta_files:
            continue

        pb_sample = pd.read_csv(matrix_files[0], index_col=0)
        meta_df = pd.read_csv(meta_files[0], index_col=0)

        meta_df.index = meta_df.index.astype(str)

        common_samples = [s for s in pb_sample.columns if s in meta_df.index]
        if len(common_samples) == 0:
            continue

        pb_sample = pb_sample[common_samples]
        meta_df = meta_df.loc[common_samples]

        results.append((cell_type_dir.name, cell_type_dir, pb_sample, meta_df))

    return results


def main():
    seed_everything(19960915)

    path = Path(
        "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium"
    )

    logs_dir = path / "logs"
    logger = setup_logger(log_dir=logs_dir, log_name="pseudobulk_celltype_PCA")

    input_dir = path / "output" / "pb" / "pb_data_celltype"
    out_dir = path / "output" / "pb" / "pb_plots_celltype"
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_diagnosis = ["IPF", "LUNG_CANCER"]
    subset_suffix = "v".join(subset_diagnosis).replace(" ", "_")
    out_dir = out_dir / subset_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_palette = {
        "IPF": "#6A7FB5",
        "LUNG_CANCER": "#B07D4A",
    }

    y_metrics = ["n_cells", "total_counts", "mean_transcripts"]
    group_cols = ["condition", "diagnosis", "timepoint_label", "treatment_arm"]
    col_list = ["diagnosis"]

    logger.info("Loading saved pseudobulk outputs...")
    results = load_celltype_results(input_dir)
    logger.info(f"Found {len(results)} cell type folders with saved outputs.")

    for cell_type, cell_type_dir, pb_sample, meta_df in results:
        logger.info(f"Processing cell type: {cell_type}")

        # -------------------------
        # Metric plots
        # -------------------------
        for x in group_cols:
            if x not in meta_df.columns:
                continue

            for y in y_metrics:
                if y not in meta_df.columns:
                    continue

                fig = plot_metric(meta_df, x, y, cell_type)
                fig.savefig(
                    cell_type_dir / f"{safe_name(cell_type)}_{y}_by_{x}.pdf",
                    bbox_inches="tight",
                )
                plt.close(fig)

        # -------------------------
        # PCA
        # -------------------------
        logger.info("Running PCA pipeline")

        X = pb_sample.T.copy()

        # filter low-abundance genes
        min_count = 10
        X = X.loc[:, X.sum(axis=0) > min_count]

        logger.info(f"Filtered genes: {X.shape[1]}")

        X = np.log1p(X)
        X_scaled = sklearn.preprocessing.StandardScaler().fit_transform(X)

        pca = sklearn.decomposition.PCA(n_components=4)
        pca_result = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame(
            pca_result,
            columns=["PC1", "PC2", "PC3", "PC4"],
            index=X.index,
        )

        pca_df = pca_df.join(meta_df)

        if subset_diagnosis:
            pca_df = pca_df[pca_df["diagnosis"].isin(subset_diagnosis)]

        sns.set_style("white")

        # -------------------------
        # PCA plots
        # -------------------------
        for col in col_list:
            if col not in pca_df.columns:
                continue

            fig, ax = plt.subplots(figsize=(6, 5))

            is_continuous = pd.api.types.is_numeric_dtype(pca_df[col])

            if is_continuous:
                scatter = ax.scatter(
                    pca_df["PC1"],
                    pca_df["PC2"],
                    c=pca_df[col],
                    cmap="viridis",
                    s=50,
                )
                fig.colorbar(scatter, ax=ax, label=col)

            else:
                groups = pca_df[col].dropna().unique()

                sns.scatterplot(
                    data=pca_df,
                    x="PC1",
                    y="PC2",
                    hue=col,
                    hue_order=list(cat_palette.keys()),
                    palette=cat_palette,
                    s=50,
                    ax=ax,
                )

                for g in groups:
                    subset = pca_df[pca_df[col] == g]
                    if len(subset) >= 2:
                        color = cat_palette.get(g, "black")

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

            ax.set_title(f"PCA: {col}")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)")

            plt.tight_layout()

            fig.savefig(
                out_dir / f"{col}_{safe_name(cell_type)}_pca.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)

        # -------------------------
        # Pairplot
        # -------------------------
        for col in col_list:
            if col not in pca_df.columns:
                continue

            g = sns.pairplot(
                pca_df,
                vars=["PC1", "PC2", "PC3", "PC4"],
                hue=col,
                palette=cat_palette,
                plot_kws={"s": 50},
                corner=True,
            )

            if g._legend is not None:
                g._legend.set_title(col)
                g._legend.set_bbox_to_anchor((0.8, 0.9))
                g._legend.set_frame_on(False)

            g.figure.suptitle(f"PCA Pairplot: {col}", y=1.02)

            g.figure.tight_layout()
            g.figure.subplots_adjust(right=0.85)

            g.figure.savefig(out_dir / f"{col}_{safe_name(cell_type)}_pca_pairplot.pdf")
            plt.close(g.figure)

        logger.info("Finished PCA plots")


if __name__ == "__main__":
    main()
