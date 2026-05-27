"""Calculate clusters for cell type annotation for level 1 subsets."""

import logging
import os
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import scanpy as sc
import seaborn as sns

from utils.seed_everything import seed_everything


def S_score_G2M_score(adata, subset):
    """Calculate S and G2M scores for cell cycle analysis and plot on UMAP.

    Args:
        adata: An AnnData object containing the single-cell data.
        subset: A string indicating the subset of data being analyzed (e.g., "immune").

    Returns:
        adata: The input AnnData object with added S and G2M scores in adata.obs.
    """
    s_genes = [
        "MCM5",
        "PCNA",
        "TYMS",
        "FEN1",
        "MCM2",
        "MCM4",
        "RRM1",
        "UNG",
        "GINS2",
        "MCM6",
        "CDCA7",
        "DTL",
        "PRIM1",
        "UHRF1",
        "HELLS",
        "RFC2",
        "RPA2",
        "NASP",
        "RAD51AP1",
        "GMNN",
        "WDR76",
        "SLBP",
        "CCNE2",
        "UBR7",
        "POLD3",
        "MSH2",
        "ATAD2",
        "RAD51",
        "RRM2",
        "CDC45",
        "CDC6",
        "EXO1",
        "TIPIN",
        "DSCC1",
        "BLM",
        "CASP8AP2",
        "USP1",
        "CLSPN",
        "POLA1",
        "CHAF1B",
        "BRIP1",
        "E2F8",
    ]

    g2m_genes = [
        "HMGB2",
        "CDK1",
        "NUSAP1",
        "UBE2C",
        "BIRC5",
        "TPX2",
        "TOP2A",
        "NDC80",
        "CKS2",
        "NUF2",
        "CKS1B",
        "MKI67",
        "TMPO",
        "CENPF",
        "TACC3",
        "FAM64A",
        "SMC4",
        "CCNB2",
        "CKAP2L",
        "CKAP2",
        "AURKB",
        "BUB1",
        "KIF11",
        "ANP32E",
        "TUBB4B",
        "GTSE1",
        "KIF20B",
        "HJURP",
        "CDC20",
        "TTK",
        "CDC25C",
        "KIF2C",
        "RANGAP1",
        "NCAPD2",
        "DLGAP5",
        "CDCA3",
        "HMMR",
        "AURKA",
        "PSRC1",
        "ANLN",
        "LBR",
        "CKAP5",
        "CENPE",
        "CTCF",
        "NEK2",
        "G2E3",
        "GAS2L3",
        "CBX5",
        "CENPA",
    ]

    sc.tl.score_genes_cell_cycle(
        adata,
        s_genes=s_genes,
        g2m_genes=g2m_genes,
    )

    sc.pl.umap(
        adata,
        color=["S_score", "G2M_score"],
        frameon=False,
        cmap=cmap_blue,
        save=f"_{subset}_s_score_G2M.pdf",
    )


def subcluster_leiden_analysis(
    adata,
    subset: str,
    subset_dir: Path,
    fig_dir_name: str = "figs",
    res_list: Iterable[float] = (0.1, 0.3, 0.5, 0.8, 1.0),
    neighbors_key: str | None = None,
    umap_key: str | None = None,
    n_dotplot_genes: int = 4,
    n_top_genes_export: int = 10,
    cmap_dotplot=None,
    logger: logging.Logger | None = None,
):
    """Perform Leiden subclustering analysis for multiple resolutions.

    For each resolution:
        - Calculates Leiden using specified neighbors_key (if provided)
        - Plots using specified umap_key (if provided)
        - Assigns cluster colors
        - Computes marker genes (Wilcoxon)
        - Plots dotplot
        - Exports full and top marker genes per cluster.

    Args:
        adata: AnnData object containing the single-cell data.
        subset: String prefix for Leiden clustering keys (e.g., "airway_epithelium").
        subset_dir: Path to directory where results for this subset will be saved.
        res_list: Iterable of resolution values to compute Leiden clustering for.
        neighbors_key : str, optional
            Key in adata.uns specifying which neighbor graph to use.
        umap_key : str, optional
            Key in adata.obsm specifying which UMAP embedding to plot.
        fig_dir_name : str, default "figs"
            Base name for figure directories (will be appended with resolution).
        n_dotplot_genes : int, default 4
            Number of top genes to show in the dotplot.
        n_top_genes_export : int, default 10
            Number of top genes to export for each cluster based on scores.
        cmap_dotplot : matplotlib colormap, optional
            Colormap to use for the dotplot.
        logger : logging.Logger, optional
            Logger for logging messages. If None, no logging will be performed.

    Returns:
        None. Modifies adata in place and saves figures/files to subset_dir.
    """
    for res in res_list:
        if logger:
            logger.info(f"Resolution: {res}")

        # Make directory for this resolution
        res_dir = subset_dir / f"leiden_resolution_{res}"

        file_dir = res_dir / f"files_{res}"
        file_dir.mkdir(parents=True, exist_ok=True)

        fig_dir = res_dir / f"{fig_dir_name}_{res}"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Set figure directory
        old_figdir = sc.settings.figdir
        sc.settings.figdir = fig_dir

        subset_key = f"{subset}_{res}"

        # Calculate Leiden clusters if not already present
        if subset_key not in adata.obs:
            sc.tl.leiden(
                adata,
                resolution=res,
                key_added=subset_key,
                flavor="igraph",
                n_iterations=2,
                neighbors_key=neighbors_key,
            )

        # Ensure categorical dtype
        if not pd.api.types.is_categorical_dtype(adata.obs[subset_key]):
            adata.obs[subset_key] = adata.obs[subset_key].astype("category")

        categories = adata.obs[subset_key].cat.categories
        n_clusters = len(categories)

        if logger:
            logger.info(f"Resolution {res}: {n_clusters} clusters")
            logger.info(
                f"Cluster distribution: "
                f"{adata.obs[subset_key].value_counts().sort_index().to_dict()}"
            )

        # Assign colors
        palette = sns.color_palette("hls", n_clusters).as_hex()
        adata.uns[f"{subset_key}_colors"] = list(palette)

        # Plot UMAP (default or custom embedding)
        if umap_key is None:
            sc.pl.umap(
                adata,
                color=subset_key,
                frameon=False,
                save=f"_{subset_key}.pdf",
            )
        else:
            if umap_key not in adata.obsm:
                raise KeyError(f"{umap_key} not found in adata.obsm")

            sc.pl.embedding(
                adata,
                basis=umap_key,
                color=subset_key,
                frameon=False,
                save=f"_{subset_key}.pdf",
            )

        rank_key = f"rank_genes_leiden_{subset_key}"

        if logger:
            logger.info("Running rank_genes_groups")

        sc.tl.rank_genes_groups(
            adata,
            groupby=subset_key,
            method="wilcoxon",
            key_added=rank_key,
        )

        sc.tl.dendrogram(adata, groupby=subset_key)

        sc.pl.rank_genes_groups_dotplot(
            adata,
            groupby=subset_key,
            standard_scale="var",
            n_genes=n_dotplot_genes,
            key=rank_key,
            cmap=cmap_dotplot,
            save=f"_{subset_key}.pdf",
        )

        markers = sc.get.rank_genes_groups_df(
            adata,
            key=rank_key,
            group=None,
        )

        if logger:
            logger.info(f"Total marker genes found: {len(markers)}")

        top_genes = (
            markers.sort_values("scores", ascending=False)
            .groupby("group", as_index=False)
            .head(n_top_genes_export)
        )

        for cluster in categories:
            cluster_str = str(cluster)

            cluster_markers = markers[markers["group"] == cluster].sort_values(
                "logfoldchanges", ascending=False
            )

            cluster_file = (
                file_dir
                / "marker_lfc"
                / f"{subset}_leiden_{res}_markers_cluster_{cluster_str}.csv"
            )
            cluster_file.parent.mkdir(parents=True, exist_ok=True)
            cluster_markers.to_csv(cluster_file, index=False)

            cluster_top = top_genes[top_genes["group"] == cluster]

            top_file = (
                file_dir
                / "top_scores"
                / f"{subset}_leiden_{res}_top_dotplot_genes_cluster_{cluster_str}.csv"
            )
            top_file.parent.mkdir(parents=True, exist_ok=True)
            cluster_top.to_csv(top_file, index=False)

            if logger:
                logger.info(
                    f"Saved cluster {cluster_str}: {len(cluster_markers)} markers"
                )

        # Restore previous figdir
        sc.settings.figdir = old_figdir


def map_clusters_to_annotations(
    adata,
    subset: str,
    chosen_resolution: float,
    annotation_dict: dict,
    annotation_level: str,
    logger: logging.Logger | None = None,
) -> None:
    """Map Leiden clusters at a chosen resolution to cell type annotations.

    This function maps cluster labels from a specified Leiden resolution
    to user-defined annotation labels and stores the result in a new
    categorical column in ``adata.obs``.

    Args:
        adata: AnnData object containing clustering results.
        subset: Prefix used for Leiden clustering keys (e.g., "airway_epithelium").
        chosen_resolution: Resolution value corresponding to the
            clustering column (e.g., 0.5).
        annotation_dict: Dictionary mapping cluster labels to
            annotation names (e.g., {"0": "Naive", "1": "Memory"}).
        annotation_level: Name of the new column in ``adata.obs``
            where annotations will be stored.
        logger: Optional logger for reporting progress and warnings.

    Returns:
        None. The function modifies ``adata.obs`` in-place.

    Raises:
        None explicitly. Logs warnings if the cluster key is missing,
        the annotation dictionary is empty, or some clusters are not mapped.
    """
    # Resolve the cluster column name based on subset and chosen resolution
    rename_subset_key = chosen_resolution_name

    # Check cluster column exists
    if rename_subset_key not in adata.obs:
        if logger:
            logger.warning(
                "%s not found in adata.obs. Cannot map clusters.",
                rename_subset_key,
            )
        return

    # Check mapping provided
    if not annotation_dict:
        if logger:
            logger.warning("annotation_dict is empty or None. Cannot map clusters.")
        return

    if logger:
        logger.info("Mapping clusters to cell type annotations.")

    mapped = adata.obs[rename_subset_key].map(annotation_dict)

    # Check for unmapped clusters
    if mapped.isna().any():
        missing = adata.obs.loc[mapped.isna(), rename_subset_key].unique().tolist()
        if logger:
            logger.warning("Some clusters were not mapped: %s", missing)

    # Store as categorical
    adata.obs[annotation_level] = mapped.astype("category")

    if logger:
        distribution = adata.obs[annotation_level].value_counts().to_dict()
        logger.info(
            "Annotation mapping completed. Cell type distribution: %s",
            distribution,
        )


def viz_annotation_level(
    adata,
    annotation_level: str,
    fig_dir: Path,
    file_dir: Path,
    subset: str,
    cmap,
    logger,
    condition_key: str = "condition",
    roi_key: str = "ROI",
    cell_id_key: str = "cell_id",
    n_top_genes: int = 5,
) -> None:
    """Visualize a given annotation level in AnnData.

    Args:
        adata: AnnData
            Annotated data matrix.
        annotation_level: str
            Column name in adata.obs containing annotations.
        fig_dir: Path
            Directory to save figures.
        file_dir: Path
            Directory to save output CSV files.
        subset: str
            Name used as prefix for output files.
        cmap: matplotlib colormap
            Colormap for dotplot.
        logger: logging.Logger
            Logger instance.
        condition_key: str, optional
        Column name for condition (default: "condition").
        roi_key : str, optional
            Column name for ROI (default: "ROI").
        cell_id_key : str, optional
            Column name for cell IDs (default: "cell_id").
        n_top_genes : int, optional
            Number of marker genes to plot (default: 5).
    """
    if annotation_level not in adata.obs:
        logger.warning(
            f"Annotation column '{annotation_level}' was not found. "
            "Skipping annotation-dependent outputs."
        )
        return

    logger.info(f"Annotation column '{annotation_level}' found. Processing...")

    # Set figure directory
    sc.settings.figdir = fig_dir

    # UMAP visualization
    sc.pl.umap(
        adata,
        color=annotation_level,
        legend_loc="right margin",
        legend_fontsize=14,
        frameon=False,
        ncols=2,
        wspace=0.4,
        save=f"_{annotation_level}_umap.pdf",
    )

    # Rank genes
    rank_key = f"rank_genes_{annotation_level}"
    sc.tl.rank_genes_groups(
        adata,
        groupby=annotation_level,
        method="wilcoxon",
        key_added=rank_key,
    )

    # Dendrogram
    dendro_key = f"dendrogram_{annotation_level}"
    sc.tl.dendrogram(
        adata,
        groupby=annotation_level,
        key_added=dendro_key,
    )

    # Dotplot
    sc.pl.rank_genes_groups_dotplot(
        adata,
        groupby=annotation_level,
        dendrogram=dendro_key,
        standard_scale="var",
        n_genes=n_top_genes,
        key=rank_key,
        cmap=cmap,
        save=f"_{annotation_level}_dotplot.pdf",
    )

    # Cell type counts
    counts = adata.obs[annotation_level].value_counts().to_frame(name="count")
    counts.to_csv(file_dir / f"_{annotation_level}_{subset}_celltype_counts.csv")

    # Counts per condition and ROI
    if condition_key in adata.obs and roi_key in adata.obs:
        counts_condition = pd.crosstab(
            adata.obs[annotation_level],
            [adata.obs[condition_key], adata.obs[roi_key]],
        )
        counts_condition.to_csv(
            file_dir
            / f"_{annotation_level}_{subset}_celltype_counts_per_condition_ROI.csv"
        )
    else:
        logger.warning("Condition or ROI columns missing; skipping crosstab.")

    # Export annotations
    if cell_id_key in adata.obs:
        df = adata.obs[[annotation_level, cell_id_key]].copy()
        df.columns = ["cell_annotation", "cell_id"]
        df.to_csv(
            file_dir / f"_{annotation_level}_{subset}_annotations.csv", index=False
        )
    else:
        logger.warning(f"'{cell_id_key}' not found in adata.obs; skipping export.")

    logger.info(f"Finished processing annotation level '{annotation_level}'.")


def recalc_umap(
    adata,
    n_pca_comps: int = 50,
    use_highly_variable: bool = True,
    n_neighbors: int = 15,
    n_pcs: int = 20,
    neighbors_key: str = "neighbors_umap_recalc",
    umap_key: str = "umap_recalc",
    leiden_key: str = "leiden_recalc",
) -> None:
    """Ensure PCA, neighbors, UMAP, and Leiden are computed from a recalculated PCA.

    PCA is computed and stored in `adata.obsm["X_pca_recalc"]`.
    Neighbor graph, UMAP, and Leiden clustering are computed using
    this representation only.

    Args:
        adata: Annotated data matrix.
        n_pca_comps: Number of principal components.
        use_highly_variable: Whether to use HVGs for PCA.
        n_neighbors: Number of neighbors.
        n_pcs: Number of PCs used for neighbors.
        neighbors_key: Key for neighbors graph.
        umap_key: Key for UMAP embedding.
        leiden_key: Key for Leiden clustering.

    Returns:
        None. Operates in place.
    """
    pca_rep = "X_pca_recalc"

    # PCA
    if pca_rep not in adata.obsm:
        logger.info("Running recalculated PCA...")
        sc.pp.pca(
            adata,
            n_comps=n_pca_comps,
            use_highly_variable=use_highly_variable,
        )
        adata.obsm[pca_rep] = adata.obsm["X_pca"].copy()
    else:
        logger.info("Recalculated PCA already present.")

    # Neighbors (using recalculated PCA)
    if neighbors_key not in adata.uns:
        logger.info("Computing neighbors from recalculated PCA...")
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            use_rep=pca_rep,
            key_added=neighbors_key,
        )
    else:
        logger.info("Neighbors already present.")

    # UMAP
    umap_obsm_key = f"X_{umap_key}"
    if umap_obsm_key not in adata.obsm:
        logger.info("Computing UMAP...")
        sc.tl.umap(
            adata,
            neighbors_key=neighbors_key,
            key_added=umap_key,
        )
    else:
        logger.info("UMAP already present.")


# Set random seed for reproducibility
seed_everything(19960915)

# Set variables
h5ad_file = os.getenv("H5AD_FILE")
subset = os.getenv("SUBSET")
annotation_level = "level_1"
res_list = [0.3, 0.5, 0.8, 1.0, 1.5]

# Resolution to use for mapping clusters to annotations
resolution = 0.5
# chosen_resolution_name = f"Immune_cells_{resolution}"
chosen_resolution_name = f"{subset}_{resolution}"
# Annotate clusters based on marker genes and plot UMAP
level_2_annotation_dict = {
    "0": "Ciliated cells",
    "1": "Ciliated cells",
    "2": "Goblet cells",
    "3": "Goblet cells",
    "4": "Goblet cells",
    "5": "Unknown",
    "6": "Ciliated cells",
    "7": "Basal cells",
    "8": "Basal cells",
    "9": "Basal cells",
    "10": "Secretory epithelial cells",
    "11": "Proliferating Basal cells",
}
level_3_annotation_dict = {
    "0": "Ciliated cells 1",
    "1": "Ciliated cells 2",
    "2": "Goblet cells 1",
    "3": "Goblet cells 2",
    "4": "Goblet cells 3",
    "5": "Unknown - Stromal cells",
    "6": "Ciliated cells 3",
    "7": "Basal cells 1",
    "8": "Basal cells 2",
    "9": "Basal cells 3",
    "10": "Secretory epithelial cells",
    "11": "Proliferating Basal cells",
}

# Define annotation column names
annotation_level_1 = "level_1"
annotation_level_2 = "level_2"
annotation_level_3 = "level_3"

# Read directory from environment variable
celltype_subset_dir = os.getenv("CELLTYPE_SUBSET_DIR")
if not celltype_subset_dir:
    raise ValueError("CELLTYPE_SUBSET_DIR environment variable must be set")
module_dir = Path(celltype_subset_dir)

subset_dir = module_dir / subset
subset_dir.mkdir(parents=True, exist_ok=True)

# Save figures
fig_dir = subset_dir / "figs"
fig_dir.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = fig_dir

# Save files
file_dir = subset_dir / "files"
file_dir.mkdir(parents=True, exist_ok=True)


# Set up logging
log_file = subset_dir / f"celltype_level_1_{subset}.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Set colors
cmap_blue = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
color_palette_level_1 = sns.color_palette("hls", 12)

# Load data
logger.info(f"Loading data from {module_dir / f'{h5ad_file}'}")
adata = sc.read_h5ad(module_dir / f"{h5ad_file}")
logger.info(f"Data loaded successfully. Shape: {adata.shape}")

# Check available columns and log cell type information
logger.info(f"Available columns in adata.obs: {list(adata.obs.columns)}")

# UMAP plot colored by manual annotation (i.e coarse clustering)
sc.pl.umap(
    adata,
    color=annotation_level,
    legend_loc="right margin",
    legend_fontsize=14,
    frameon=False,
    ncols=2,
    wspace=0.4,
    save=f"_{annotation_level_1}.pdf",
)

# Look at proliferative score (S and G2M)
S_score_G2M_score(adata, subset)

logger.info(f"Starting Level 1 cell type annotation for subset: {subset}")
logger.info(f"Output directory: {subset_dir}")

# STEP 1: Calculate leiden clusters for cell type
# Calculate Leiden clusters, marker genes, and UMAP plots for multiple resolutions
subcluster_leiden_analysis(
    adata=adata,
    subset=subset,
    subset_dir=subset_dir,
    fig_dir_name="figs",
    res_list=res_list,
    n_dotplot_genes=5,
    n_top_genes_export=10,
    cmap_dotplot=cmap_blue,
    logger=logger,
)


# # STEP 2: Recalculate UMAP using top 2000 variable genes for better
# recalc_umap(
#     adata,
#     n_pca_comps=50,
#     use_highly_variable=True,
#     n_neighbors=15,
#     n_pcs=20,
#     neighbors_key="neighbors_umap_recalc",
#     umap_key="umap_recalc",
# )

# # Re-run Leiden clustering and marker gene analysis using the recalculated UMAP graph
# subcluster_leiden_analysis(
#     adata=adata,
#     subset=subset,
#     subset_dir=subset_dir,
#     fig_dir_name="figs_recalc",
#     res_list=res_list,
#     neighbors_key="neighbors_umap_recalc",
#     umap_key="umap_recalc",
#     n_dotplot_genes=5,
#     n_top_genes_export=10,
#     cmap_dotplot=cmap_blue,
#     logger=logger,
# )

# STEP 3: Map Leiden clusters to annotation_level_2 based on marker genes
# and save in adata.obs

# # Map level 2 annotations
# map_clusters_to_annotations(
#     adata,
#     subset=subset,
#     chosen_resolution=chosen_resolution_name,
#     annotation_dict=level_2_annotation_dict,
#     annotation_level=annotation_level_2,
#     logger=logger,
# )

# # Map level 3 annotations
# map_clusters_to_annotations(
#     adata,
#     subset=subset,
#     chosen_resolution=chosen_resolution_name,
#     annotation_dict=level_3_annotation_dict,
#     annotation_level=annotation_level_3,
#     logger=logger,
# )

# # Visualize annotation levels
# viz_annotation_level(
#     adata=adata,
#     annotation_level=annotation_level_2,
#     fig_dir=fig_dir,
#     file_dir=file_dir,
#     subset=subset,
#     cmap=cmap_blue,
#     logger=logger,
# )

# viz_annotation_level(
#     adata=adata,
#     annotation_level=annotation_level_3,
#     fig_dir=fig_dir,
#     file_dir=file_dir,
#     subset=subset,
#     cmap=cmap_blue,
#     logger=logger,
# )

# Save the annotated data
output_file = module_dir / f"adata_subset_{subset}.h5ad"
logger.info(f"Saving annotated data to {output_file}")
adata.write_h5ad(output_file)
logger.info("Main analysis data saved successfully")
logger.info(f"Final data shape: {adata.shape}")

logger.info(f"Analysis completed successfully for {subset}")
print("Done!")
