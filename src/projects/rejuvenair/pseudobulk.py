"""Calculate pseudobulk differential expression for Xenium data."""

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch

# import torch
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


# Functions
def seed_everything(seed: int):
    """Set random seed on every random module for reproducibility.

    Args:
        seed: The seed value to set for random number generation.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    else:
        pass


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


def volcano_plot(
    de_results: pd.DataFrame,
    lfc_threshold: int = 5,
    padj_threshold: float = 0.05,
    use_padj=True,
    title="Volcano Plot: Differential Expression",
):
    """Volcano plot for differentially expressed genes.

    Args:
        de_results : DeseqStats or pd.DataFrame DESeq2 results object or DataFrame.
        lfc_threshold : float
        Absolute log2 fold change threshold to highlight genes.
        padj_threshold : float
        Adjusted p-value threshold to highlight genes.
        use_padj : bool
        If True, use adjusted p-value (padj)
        title : str
        Plot title.

    Returns:
        Plot.
    """
    # Convert to DataFrame if DeseqStats
    if hasattr(de_results, "results_df"):
        df = de_results.results_df.copy()
    else:
        df = de_results.copy()

    # Determine p-value column
    p_col = "padj"
    if p_col not in df.columns:
        raise ValueError(f"{p_col} column not found in DE results")

    # Fill NaNs
    df[p_col] = df[p_col].fillna(1)

    # Determine gene categories
    up = (df["log2FoldChange"] > lfc_threshold) & (df[p_col] < padj_threshold)
    down = (df["log2FoldChange"] < -lfc_threshold) & (df[p_col] < padj_threshold)
    sig_small_lfc = (abs(df["log2FoldChange"]) <= lfc_threshold) & (
        df[p_col] < padj_threshold
    )
    neutral = ~(up | down | sig_small_lfc)

    plt.figure(figsize=(10, 12))

    # Plot categories
    plt.scatter(
        df.loc[neutral, "log2FoldChange"],
        -np.log10(df.loc[neutral, p_col]),
        color="grey",
        alpha=0.6,
        s=10,
        fontsize=8,
    )
    plt.scatter(
        df.loc[up, "log2FoldChange"],
        -np.log10(df.loc[up, p_col]),
        color="#8ab184",
        alpha=0.8,
        label=f"Upregulated (LFC > {lfc_threshold})",
        s=15,
    )
    plt.scatter(
        df.loc[down, "log2FoldChange"],
        -np.log10(df.loc[down, p_col]),
        color="#BC3C29",
        alpha=0.8,
        label=f"Downregulated (LFC < -{lfc_threshold})",
        s=15,
    )
    plt.scatter(
        df.loc[sig_small_lfc, "log2FoldChange"],
        -np.log10(df.loc[sig_small_lfc, p_col]),
        color="#6F99AD",
        alpha=0.8,
        label=f"Significant, |LFC| ≤ {lfc_threshold}",
        s=15,
    )

    # Label only up/down genes
    for _, row in df.loc[up | down].iterrows():
        plt.text(
            row["log2FoldChange"],
            -np.log10(row[p_col]),
            row.name,
            fontsize=12,
            ha="right",
        )

    # Add vertical dotted lines for LFC threshold
    plt.axvline(lfc_threshold, color="black", linestyle=":", linewidth=1)
    plt.axvline(-lfc_threshold, color="black", linestyle=":", linewidth=1)

    # Add horizontal dotted line for p-value threshold
    plt.axhline(-np.log10(padj_threshold), color="black", linestyle=":", linewidth=1)

    # Labels
    plt.xlabel("log2 Fold Change (COPD vs IPF)")
    plt.ylabel(f"-log10({p_col})")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


# Set random seed for reproducibility
seed_everything(19960915)

# Set up logger
wd = "/rds/general/user/sep22/home/Projects/AirScape/HPC_jobs/rejuvenair/"
logs_dir = Path(wd) / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logger(log_dir=logs_dir, log_name="pseudobulk_DE")

# Set variables
level_1 = "level_1"
level_2 = "level_2"

# set subset
cell_type_subset = "Stromal cells"

# design factors for DESeq2
design_factors = ["condition"]
contrast = ["condition", "COPD", "IPF"]

# Set directory
dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/2026-03-27_analysis_run/"
)

# Set figure directory
folder_name = "pseudobulk_DE_results"
fig_path = f"project_analysis/RejuvenAir_MICAIII/{folder_name}"

fig_dir = dir / fig_path
os.makedirs(fig_dir, exist_ok=True)

# set fig dir for plots to save to
sc.settings.figdir = fig_dir

# Set colors
cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
color_palette_level_1 = sns.color_palette("hls", 12)

# Load data
logger.info("Loading data...")
adata = sc.read_h5ad(dir / "subset_adata/adata_subset_COPD_MICA_clean.h5ad")

# Subset on specific cell type
adata = adata[adata.obs["level_1_corrected"] == cell_type_subset].copy()
fig_dir = dir / fig_path / cell_type_subset.replace(" ", "_")
os.makedirs(fig_dir, exist_ok=True)
sc.settings.figdir = fig_dir  # set fig dir for plots to save to


# Pseudobulk and PCA on all samples
logger.info("Calculating pseudobulk and PCA for all samples (all cell types)")
# TODO: Plot PCA of pseudobulk samples to check for batch effects, etc. before DE analysis

# Subset to only desired annotation
logger.info(f"Subsetting on {cell_type_subset} annotations")
logger.info(f"Calculating DEG for each cell type in {level_2} annotations")

unique_cell_types = adata.obs[level_2].unique().tolist()
for cell_type in unique_cell_types:
    logger.info("Processing cell type:", cell_type)
    cell_subset = adata[adata.obs[level_2] == cell_type]

    pbs = []  # pseudobulk sample
    for sample in cell_subset.obs["ROI"].unique():
        sample_cell_subset = cell_subset[cell_subset.obs["ROI"] == sample].copy()
        sample_cell_subset.X = sample_cell_subset.layers[
            "counts"
        ]  # Use raw data rather than normalized log transformed data
        # Create new adata for each sample
        rep_data = sc.AnnData(
            X=sample_cell_subset.X.sum(axis=0), var=sample_cell_subset.var[[]]
        )

        rep_data.obs_names = [sample]  # set obs name to the sample
        rep_data.obs[design_factors] = sample_cell_subset.obs[design_factors].iloc[0]

        # Add to psuedobulk
        pbs.append(rep_data)

        # Concatenate all pseudobulk samples
        psuedobulk = sc.concat(pbs)

        # Differential expression
        counts = pd.DataFrame(
            psuedobulk.X, columns=psuedobulk.var_names
        )  # need to do this to pass var names

        # Create DESeq2 dataset
        dds = DeseqDataSet(
            counts=counts, metadata=psuedobulk.obs, design_factors=design_factors
        )

        # Filter to remove genes not found in at least one sample
        sc.pp.filter_genes(dds, min_cells=1)
        print(f"Number of genes after filtering: {dds.X.shape[1]}")

        # Initialize DESeq2
        dds.deseq2()

        # Get DE results
        de_results = DeseqStats(dds, n_cpus=8, contrast=contrast)
        de_results.summary()
        results_df = de_results.results_df  # Get results dataframe
        results_df.to_csv(f"pseudobulk_DE_results_{cell_type}.csv")  # Save DE results

        # Plot PCA of psuedobulk samples
        sc.tl.pca(dds)
        sc.pl.pca(
            dds,
            color="condition",
            size=200,
            title=f"PCA of psuedobulk samples - {cell_type}",
            save=f"_pseudobulk_PCA_{cell_type}.pdf",
        )

        # Plot volcano plot
        volcano_plot(de_results, lfc_threshold=2, padj_threshold=0.05, use_padj=True)
        plt.savefig(f"volcano_plot_pseudobulk_{cell_type}.pdf")
        plt.close()

logger.info("Pseudobulk differential expression complete.")
