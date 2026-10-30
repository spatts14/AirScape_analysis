"""Cell-cell communication (ligand-receptor) analysis: IPF vs PM08.

Four complementary analyses, all grouped by adata.obs["level_2"]:

  Part A - Per-condition LIANA consensus (rank_aggregate run separately for
           IPF and PM08). Answers: which interactions are strong/specific in
           each disease, and how does their magnitude (lr_means) shift?
           Descriptive; does not account for sample-to-sample variation.

  Part B - Sample-aware differential analysis. Pseudobulk (ROI x level_2),
           PyDESeq2 IPF vs PM08 (PM08 = reference) per cell type, then LIANA
           maps the DE statistics onto ligand-receptor pairs
           (li.multi.df_to_lr). Use this part for statistical claims.

  Part C - CrossTalkeR: cell types ranked by PageRank on the communication
           graph, compared between conditions (+ Fisher / Mann-Whitney tests).

  Part D - Spatial: local co-expression (LIANA bivariate) of the top
           differential L-R pairs from Part B, mapped onto each ROI.

Requirements (tested with liana 1.10, pydeseq2 0.5.4, scanpy 1.11,
pycrosstalker 2.1.8): see setup_venv_RL.sh

Outputs go to OUTDIR = <INPUT_DIR>/ccc/IPFvPM08 (CSV tables + PDF/PNG figures).
"""

from __future__ import annotations

import gc
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import liana as li
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pycrosstalker as pyct
import pycrosstalker.plots as pyct_plots
import scanpy as sc
import zarr
from liana.mt import rank_aggregate
from matplotlib.backends.backend_pdf import PdfPages
from plotnine import ggtitle, labs, save_as_pdf_pages
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scanpy.plotting import palettes
from scipy import sparse
from scipy.stats import mannwhitneyu

try:
    from anndata.io import read_elem, sparse_dataset
except ImportError:  # older anndata
    from anndata.experimental import read_elem, sparse_dataset

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIG - edit these
# =============================================================================
INPUT_DIR = Path("/rds/general/user/sep22/ephemeral/")
INPUT_ZARR = INPUT_DIR / "adata_with_metadata.zarr"
OUTDIR = INPUT_DIR / "ccc" / "COPDvMICA"

CONDITION_KEY = "condition"
CONDITIONS = ["COPD", "MICA"]
REFERENCE = "MICA"  # baseline: red = higher in COPD, blue = lower in COPD
TEST = "COPD"
CMP = f"{TEST}_x_{REFERENCE}"  # CrossTalkeR comparison name, e.g. "COPD_x_MICA"

GROUP_KEY = "level_2"  # cell types used as sources/targets
SAMPLE_KEY = "ROI"  # replicate unit for pseudobulk (Part B)
DROP_ROIS = ["PM08_159"]  #! update to drop anything but V1
DROP_CELL_TYPES = [
    "Alveolar fibroblasts",
    "Alveolar fibroblasts (collagen high)",
    "AT1 cells",
    "AT2 cells",
    "Lipid-associated macrophages",
    "Airway/Alveolar macrophages",
    "Proliferating AT2 cells",
]

# Where raw counts live in the zarr store. None = X already holds raw counts.
COUNTS_LAYER = "counts"
LOAD_CHUNK = 20000  # rows read at a time when the matrix is stored dense

RESOURCE = "consensus"  # "mouseconsensus" for mouse data
EXPR_PROP = 0.1  # min fraction of cells expressing ligand/receptor
MIN_CELLS_PER_GROUP = 10  # LIANA: min cells per level_2 group
N_PERMS = 1000  # permutations for specificity p-values

# Pseudobulk / DESeq2 (Part B)
PB_MIN_CELLS = 10  # min cells for a ROI x level_2 pseudobulk sample
PB_MIN_SAMPLES_PER_COND = 2  # skip a cell type with fewer replicates per condition
GENE_MIN_COUNTS = 10  # gene filter for DESeq2

SIG_RANK = 0.05  # magnitude/specificity rank threshold (Part A)
PADJ = 0.05  # adjusted p threshold (Part B)
TOP_N = 30  # interactions shown in the top-N plots
DOTPLOT_PVAL = 0.05  # A6: keep interactions with cellphone_pvals <= this
DOTPLOT_TOP_N = 30  # A6: max interactions per dotplot (by lr_means); None = all

# Spatial (Part D) - coordinates in adata.obsm["spatial"], in um
SPATIAL_KEY = "spatial"
SPATIAL_BANDWIDTH = 100  # um; Gaussian kernel width for neighborhoods
SPATIAL_CUTOFF = 0.1  # drop neighbor weights below this
SPATIAL_TOP_PAIRS = 10  # top differential L-R pairs mapped (half up, half down)
SPATIAL_NZ_PROP = 0.05  # min fraction of cells expressing ligand & receptor per ROI
SPATIAL_POINT_SIZE = 0.5  # scatter point size on tissue maps
SPATIAL_MIN_CELLS = 50  # skip ROIs with fewer cells

# Plot styling
UP_COLOR = "#c0392b"  # higher in TEST
DOWN_COLOR = "#2c6fbb"  # higher in REFERENCE
COND_COLORS = {TEST: UP_COLOR, REFERENCE: DOWN_COLOR}
CMAP_SEQ = "Blues"  # non-negative values (counts, lr_means, local scores)
CMAP_DIV = "RdBu_r"  # signed values (differences, statistics)
CTK_DIR = "C_crosstalker"
SPATIAL_DIR = "D_spatial"

# li.pl.dotplot settings shared by the two per-source dotplot families
RANK_DOTPLOT = dict(  # A0: LIANA consensus ranks (small = strong/specific)
    colour="magnitude_rank",
    size="specificity_rank",
    inverse_colour=True,
    inverse_size=True,
    orderby="magnitude_rank",
    orderby_ascending=True,
)
PVAL_DOTPLOT = dict(  # A6: CellPhoneDB-style magnitude + p-value
    colour="lr_means",
    size="cellphone_pvals",
    inverse_size=True,
    orderby="lr_means" if DOTPLOT_TOP_N else None,
    orderby_ascending=False if DOTPLOT_TOP_N else None,
)

COLORS_KEY = f"{GROUP_KEY}_colors"
LR_KEY_COLS = ["source", "target", "ligand_complex", "receptor_complex"]


# =============================================================================
# Generic helpers
# =============================================================================
def out_path(stem: str, ext: str) -> Path:
    """OUTDIR/<stem>.<ext>, creating parent directories as needed."""
    path = OUTDIR / f"{stem}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def savefig(fig, stem: str) -> None:
    """Save a matplotlib figure as PDF + PNG under OUTDIR and close it."""
    for ext in ("pdf", "png"):
        fig.savefig(out_path(stem, ext), dpi=300)
    plt.close(fig)


def save_plotnine(plot, stem: str) -> None:
    """Save a plotnine figure as PNG + PDF under OUTDIR."""
    for ext in ("png", "pdf"):
        plot.save(out_path(stem, ext), dpi=300, verbose=False, limitsize=False)


def safe_name(text) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def interaction_label(df: pd.DataFrame) -> pd.Series:
    """'ligand → receptor  (source → target)' label for each row."""
    return (
        df["ligand_complex"]
        + " → "
        + df["receptor_complex"]
        + "  ("
        + df["source"].astype(str)
        + " → "
        + df["target"].astype(str)
        + ")"
    )


def genes_in_resource(resource: pd.DataFrame, var_names) -> list[str]:
    """Genes (complex subunits split on '_') of a ligand/receptor table."""
    parts = pd.concat([resource["ligand"], resource["receptor"]])
    genes = {g for x in parts for g in str(x).split("_")}
    return sorted(genes & set(var_names))


def heatmap_figsize(n: int, ncols: int = 1) -> tuple[float, float]:
    return ncols * max(5, 0.45 * n + 2), max(4.5, 0.45 * n + 1.5)


def count_matrix(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Number of interactions per source (rows) x target (columns)."""
    return pd.crosstab(df["source"], df["target"]).reindex(
        index=groups, columns=groups, fill_value=0
    )


def source_target_heatmap(
    fig,
    ax,
    values,
    groups,
    title,
    *,
    cmap,
    vmin=None,
    vmax=None,
    cbar_label="",
    stars=None,
) -> None:
    """Source x target heatmap; `stars` is an optional boolean mask to mark."""
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    if stars is not None:
        for i, j in zip(*np.nonzero(stars)):
            ax.text(j, i, "*", ha="center", va="center", fontsize=10)
    ticks = range(len(groups))
    ax.set_xticks(ticks)
    ax.set_xticklabels(groups, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels(groups)
    ax.set_xlabel("Target (receptor)")
    ax.set_ylabel("Source (ligand)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7, label=cbar_label)


def diverging_barh(ax, labels, values) -> None:
    """Horizontal bars colored by sign (UP_COLOR > 0, DOWN_COLOR otherwise)."""
    ax.barh(
        labels, values, color=np.where(np.asarray(values) > 0, UP_COLOR, DOWN_COLOR)
    )
    ax.axvline(0, color="k", lw=0.8)


def circle_plots(adata, df, name, title, cell_types) -> None:
    """Circle plots of interaction counts between level_2 groups.

    Page 1 = all cell types; then one page per cell type showing its
    outgoing (as sender) and incoming (as receiver) interactions.
    Saved as one multi-page PDF plus a PNG of the overview.
    Edge width = number of interactions; widths are scaled within each plot,
    so compare across plots using the counts heatmaps, not edge widths.
    """
    if df.empty:
        print(f"  {name}: no interactions to plot")
        return
    kw = dict(
        groupby=GROUP_KEY,
        liana_res=df,
        pivot_mode="counts",
        figure_size=(6, 6),
        node_label_size=8,
        edge_arrow_size=12,
    )
    with PdfPages(out_path(name, "pdf")) as pdf:
        ax = li.pl.circle_plot(adata, **kw)
        ax.set_title(f"{title}\nall {GROUP_KEY} groups (n = {len(df)})")
        fig = ax.get_figure()
        fig.savefig(out_path(f"{name}_overview", "png"), dpi=300)
        pdf.savefig(fig)
        plt.close(fig)
        for cell_type in cell_types:
            for role, label in (("source", "outgoing"), ("target", "incoming")):
                sub = df[df[role] == cell_type]
                if sub.empty:
                    continue
                ax = li.pl.circle_plot(
                    adata, **{**kw, "liana_res": sub, f"{role}_labels": [cell_type]}
                )
                ax.set_title(f"{title}\n{cell_type}: {label} (n = {len(sub)})")
                fig = ax.get_figure()
                pdf.savefig(fig)
                plt.close(fig)


# =============================================================================
# 1. Load, subset, preprocess
# =============================================================================
def looks_like_counts(X) -> bool:
    vals = X[:1000].toarray() if sparse.issparse(X) else np.asarray(X[:1000])
    return bool(np.all(vals >= 0) and np.allclose(vals, np.round(vals)))


def read_group_colors(g, obs: pd.DataFrame) -> dict[str, str]:
    """Existing level_2 colors keyed by category name (survive subsetting)."""
    if "uns" not in g or COLORS_KEY not in g["uns"]:
        return {}
    colors = list(read_elem(g["uns"][COLORS_KEY]))
    labels = obs[GROUP_KEY]
    if isinstance(labels.dtype, pd.CategoricalDtype) and len(colors) == len(
        labels.cat.categories
    ):
        return dict(zip(labels.cat.categories.astype(str), colors))
    return {}


def select_cells(obs: pd.DataFrame) -> np.ndarray:
    """Boolean mask of cells to keep (decided from obs before reading X)."""
    labels = obs[GROUP_KEY]
    unlabeled = (
        labels.isna() | labels.astype(str).str.lower().isin(["nan", "none", ""])
    ).to_numpy()
    keep = (
        obs[CONDITION_KEY].isin(CONDITIONS)
        & ~obs[SAMPLE_KEY].isin(DROP_ROIS)
        & ~labels.isin(DROP_CELL_TYPES)
    ).to_numpy()
    # cells without a level_2 label break the plots
    print(f"Removing {int((keep & unlabeled).sum())} cells without a {GROUP_KEY} label")
    keep &= ~unlabeled
    print(f"Keeping {keep.sum()} of {len(keep)} cells")
    return keep


def read_counts(g, keep: np.ndarray) -> sparse.csr_matrix:
    """Raw counts of the kept cells as sparse float32 (memory-efficient).

    The full object has dense X + dense counts (~25 GB each), so rows are read
    only for kept cells and dense storage is converted in chunks.
    """
    idx = np.flatnonzero(keep)
    if COUNTS_LAYER is not None and "layers" in g and COUNTS_LAYER in g["layers"]:
        src = g["layers"][COUNTS_LAYER]
    else:
        src = g["X"]
    encoding = dict(src.attrs).get("encoding-type", "array")
    if encoding == "csr_matrix":
        print("Counts stored sparse (CSR); reading selected rows")
        counts = sparse_dataset(src)[idx].tocsr().astype(np.float32)
    elif encoding == "csc_matrix":
        print("Counts stored sparse (CSC); reading then subsetting")
        counts = read_elem(src).tocsr()[idx].astype(np.float32)
    else:
        print(f"Counts stored DENSE {src.shape}; converting to sparse in chunks")
        blocks = []
        for start in range(0, src.shape[0], LOAD_CHUNK):
            stop = min(start + LOAD_CHUNK, src.shape[0])
            rows = keep[start:stop]
            if rows.any():
                block = np.asarray(src[start:stop], dtype=np.float32)[rows]
                blocks.append(sparse.csr_matrix(block))
        counts = sparse.vstack(blocks, format="csr")
    gc.collect()
    return counts


def assign_group_colors(adata: ad.AnnData, original: dict[str, str]) -> None:
    """Keep original level_2 colors; give new categories scanpy defaults."""
    cats = adata.obs[GROUP_KEY].cat.categories
    n = len(cats)
    palette = (
        palettes.default_20
        if n <= 20
        else palettes.default_28
        if n <= 28
        else palettes.default_102
    )
    adata.uns[COLORS_KEY] = [
        original.get(c, palette[i % len(palette)]) for i, c in enumerate(cats)
    ]


def load_adata() -> ad.AnnData:
    """Load the subset needed here: obs columns, gene names, coords, counts."""
    g = zarr.open(str(INPUT_ZARR), mode="r")
    obs = read_elem(g["obs"])[[CONDITION_KEY, SAMPLE_KEY, GROUP_KEY]]
    var = read_elem(g["var"])[[]]  # gene names only
    original_colors = read_group_colors(g, obs)

    keep = select_cells(obs)
    counts = read_counts(g, keep)
    spatial = np.asarray(read_elem(g["obsm"][SPATIAL_KEY]))[keep]

    adata = ad.AnnData(
        X=counts.copy(),  # normalized in place in normalize()
        obs=obs.loc[keep].copy(),
        var=var,
        obsm={SPATIAL_KEY: spatial},
        layers={"counts": counts},  # raw counts kept for Part B
    )
    for key in (CONDITION_KEY, GROUP_KEY, SAMPLE_KEY):
        adata.obs[key] = adata.obs[key].astype(str).astype("category")
    adata.obs[CONDITION_KEY] = adata.obs[CONDITION_KEY].cat.reorder_categories(
        [REFERENCE, TEST]
    )
    assign_group_colors(adata, original_colors)
    return adata


def normalize(adata: ad.AnnData) -> None:
    """Library-size normalize + log1p X in place (raw counts stay in a layer)."""
    if not looks_like_counts(adata.layers["counts"]):
        warnings.warn(
            "The counts layer doesn't look like integer counts. "
            "DESeq2 in Part B needs raw counts."
        )
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata.uns.pop("log1p", None)  # stale flag from the saved object
    sc.pp.log1p(adata)
    adata.raw = None  # make sure LIANA uses X


def print_summary(adata: ad.AnnData) -> None:
    print(adata)
    print(f"\nCells per condition x {GROUP_KEY}:")
    print(pd.crosstab(adata.obs[GROUP_KEY], adata.obs[CONDITION_KEY]))
    print("\nROIs per condition:")
    print(adata.obs.groupby(CONDITION_KEY, observed=True)[SAMPLE_KEY].nunique())


# =============================================================================
# PART A - LIANA consensus per condition
# =============================================================================
# rank_aggregate runs CellPhoneDB, Connectome, log2FC, NATMI, SingleCellSignalR
# and CellChat-style scoring, then aggregates their ranks into
# magnitude_rank (strength) and specificity_rank (cell-type specificity).
# Run separately per condition so IPF and PM08 are not pooled.
def run_rank_aggregate(adata, condition, lr_genes) -> ad.AnnData:
    # LIANA scales (densifies) the expression matrix; restricting to genes in
    # the resource cuts memory a lot and doesn't change the results.
    adata_c = adata[adata.obs[CONDITION_KEY] == condition, lr_genes].copy()
    adata_c.layers.clear()  # counts not needed here
    adata_c.uns[COLORS_KEY] = adata.uns[COLORS_KEY]
    rank_aggregate(
        adata_c,
        groupby=GROUP_KEY,
        resource_name=RESOURCE,
        expr_prop=EXPR_PROP,
        min_cells=MIN_CELLS_PER_GROUP,
        n_perms=N_PERMS,
        use_raw=False,  # log-normalized data is in X (adata.raw is None)
        verbose=False,  # True prints a 1000-line permutation progress bar
    )
    return adata_c


def dotplots_by_source(
    res_c, cond, stem, groups, *, top_n, title_note, legend, row_filter=None, **dot_kw
) -> None:
    """One dotplot per sender cell type (source -> all targets).

    Saves <stem>/<cond>/<cell type>.png plus a multi-page <stem>_<cond>.pdf.
    """
    pages = []
    for cell_type in groups:
        sub = res_c[res_c["source"] == cell_type]
        if row_filter is not None:
            sub = sub[row_filter(sub)]
        if sub.empty:
            print(f"  {stem} {cond} / {cell_type}: no interactions as sender, skipped")
            continue
        targets = [g for g in groups if g in set(sub["target"])]
        n_int = sub[["ligand_complex", "receptor_complex"]].drop_duplicates().shape[0]
        n_show = n_int if top_n is None else min(n_int, top_n)
        plot = (
            li.pl.dotplot(
                liana_res=res_c,
                source_labels=[cell_type],
                target_labels=targets,
                filter_fun=row_filter,
                top_n=top_n,
                cmap=CMAP_SEQ,
                figure_size=(
                    max(8, 0.45 * len(targets) + 4),
                    max(5, 0.28 * n_show + 2),
                ),
                **dot_kw,
            )
            + ggtitle(f"{cond}: {cell_type} → all {GROUP_KEY} ({title_note})")
            + labs(**legend)
        )
        save_plotnine(plot, f"{stem}/{cond}/{safe_name(cell_type)}")
        pages.append(plot)
    if pages:
        save_as_pdf_pages(
            pages, filename=str(out_path(f"{stem}_{cond}", "pdf")), verbose=False
        )


def sig_circle_plot(adata_c, title, stem, source=None) -> None:
    """Circle plot of interactions with magnitude_rank <= SIG_RANK.

    Edge width = number of interactions. `source` restricts to one sender.
    """
    kw = dict(
        groupby=GROUP_KEY,
        score_key="magnitude_rank",
        inverse_score=True,
        filter_fun=lambda x: x["magnitude_rank"] <= SIG_RANK,
        pivot_mode="counts",
        figure_size=(7, 7),
    )
    if source is not None:
        kw["source_labels"] = [source]
    ax = li.pl.circle_plot(adata_c, **kw)
    ax.set_title(title)
    savefig(ax.get_figure(), stem)


def plot_condition_results(adata_c, res_c, cond, groups) -> None:
    """A0 (rank-based) and A6 (p-value based) plots for one condition."""
    rank_legend = {
        "colour": "-log10(magnitude_rank)",
        "size": "-log10(specificity_rank)",
    }

    # Overview dotplot of the top interactions
    overview = (
        li.pl.dotplot(
            liana_res=res_c,
            source_labels=[g for g in groups if g in set(res_c["source"])],
            target_labels=[g for g in groups if g in set(res_c["target"])],
            top_n=TOP_N,
            cmap=CMAP_SEQ,
            figure_size=(max(10, 0.9 * len(groups) + 4), max(6, 0.3 * TOP_N + 2)),
            **RANK_DOTPLOT,
        )
        + ggtitle(f"{cond}: top {TOP_N} interactions (rank_aggregate)")
        + labs(**rank_legend)
    )
    save_plotnine(overview, f"A0_rank_aggregate_dotplot_{cond}")

    # One dotplot per sender, rank-based (A0) and p-value-based (A6)
    dotplots_by_source(
        res_c,
        cond,
        "A0_dotplots_by_source",
        groups,
        top_n=TOP_N,
        title_note=f"top {TOP_N}",
        legend=rank_legend,
        **RANK_DOTPLOT,
    )
    dotplots_by_source(
        res_c,
        cond,
        "A6_dotplots_by_source",
        groups,
        top_n=DOTPLOT_TOP_N,
        title_note=f"cellphone_pvals ≤ {DOTPLOT_PVAL}",
        legend={"size": "-log10(p)", "colour": "lr_means"},
        row_filter=lambda x: x["cellphone_pvals"] <= DOTPLOT_PVAL,
        **PVAL_DOTPLOT,
    )

    # Circle plots: whole network, then one per sender cell type
    sig_circle_plot(
        adata_c,
        f"{cond}: # interactions (magnitude_rank ≤ {SIG_RANK})",
        f"A0_rank_aggregate_circle_{cond}",
    )
    senders = set(res_c.loc[res_c["magnitude_rank"] <= SIG_RANK, "source"])
    for cell_type in groups:
        if cell_type in senders:
            sig_circle_plot(
                adata_c,
                f"{cond}: {cell_type} (magnitude_rank ≤ {SIG_RANK})",
                f"A0_circle_by_celltype/{cond}/{safe_name(cell_type)}",
                source=cell_type,
            )


def plot_a1_significant_counts(res, groups) -> None:
    """A1: number of significant interactions per source -> target."""
    mats = {
        c: count_matrix(res[(res[CONDITION_KEY] == c) & res["sig"]], groups)
        for c in CONDITIONS
    }
    diff = mats[TEST] - mats[REFERENCE]
    fig, axes = plt.subplots(1, 3, figsize=heatmap_figsize(len(groups), 3))
    vmax = max(m.values.max() for m in mats.values()) or 1
    for ax, c in zip(axes[:2], [REFERENCE, TEST]):
        source_target_heatmap(
            fig, ax, mats[c].values, groups, f"{c}: # significant L-R",
            cmap=CMAP_SEQ, vmin=0, vmax=vmax,
        )  # fmt: skip
    lim = np.abs(diff.values).max() or 1
    source_target_heatmap(
        fig, axes[2], diff.values, groups, f"{TEST} − {REFERENCE}",
        cmap=CMAP_DIV, vmin=-lim, vmax=lim,
    )  # fmt: skip
    savefig(fig, "A1_n_significant_interactions_heatmaps")


def magnitude_shift_table(res) -> pd.DataFrame:
    """lr_means in each condition for interactions significant in either."""
    wide = res.pivot_table(
        index=LR_KEY_COLS, columns=CONDITION_KEY, values="lr_means", observed=True
    )
    sig_any = res.loc[res["sig"], LR_KEY_COLS].drop_duplicates()
    wide = wide.reset_index().merge(sig_any, how="inner")
    wide[[REFERENCE, TEST]] = wide[[REFERENCE, TEST]].fillna(0)  # not detected -> 0
    wide["delta_lr_means"] = wide[TEST] - wide[REFERENCE]
    wide["log2FC_lr_means"] = np.log2((wide[TEST] + 1e-3) / (wide[REFERENCE] + 1e-3))
    wide["label"] = interaction_label(wide)
    wide = wide.sort_values("delta_lr_means")
    wide.to_csv(out_path("A_lr_means_delta", "csv"), index=False)
    return wide


def plot_a2_shift_barplot(top) -> None:
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(top) + 1.5))
    diverging_barh(ax, top["label"], top["delta_lr_means"])
    ax.set_xlabel(f"Δ mean L-R expression ({TEST} − {REFERENCE})")
    ax.set_title(
        f"Largest L-R magnitude shifts\n"
        f"(red = higher in {TEST}, blue = higher in {REFERENCE})"
    )
    ax.tick_params(axis="y", labelsize=8)
    savefig(fig, "A2_top_delta_lr_means_barplot")


def plot_a3_shift_scatter(wide, top) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.where(top["delta_lr_means"] > 0, UP_COLOR, DOWN_COLOR)
    ax.scatter(wide[REFERENCE], wide[TEST], s=6, c="grey", alpha=0.5, linewidths=0)
    ax.scatter(top[REFERENCE], top[TEST], s=14, c=colors)
    mx = max(wide[REFERENCE].max(), wide[TEST].max())
    ax.plot([0, mx], [0, mx], "k--", lw=0.8)
    for _, r in pd.concat([top.head(8), top.tail(8)]).iterrows():
        ax.annotate(
            f"{r['ligand_complex']}→{r['receptor_complex']}",
            (r[REFERENCE], r[TEST]),
            fontsize=6,
        )
    ax.set_xlabel(f"lr_means in {REFERENCE}")
    ax.set_ylabel(f"lr_means in {TEST}")
    ax.set_title("L-R magnitude per source→target pair")
    savefig(fig, "A3_lr_means_scatter")


def plot_a4_shift_dotplot(res, top) -> None:
    """Colour = lr_means (magnitude), size = -log10(specificity_rank)."""
    res_top = res.merge(top[LR_KEY_COLS + ["label", "delta_lr_means"]], how="inner")
    order = top.sort_values("delta_lr_means")["label"].tolist()
    ypos = {label: i for i, label in enumerate(order)}
    xpos = {REFERENCE: 0, TEST: 1}

    def dot_size(rank):
        return 20 + 25 * -np.log10(rank)

    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(order) + 1.8))
    sca = ax.scatter(
        res_top[CONDITION_KEY].astype(str).map(xpos),
        res_top["label"].map(ypos),
        c=res_top["lr_means"],
        cmap=CMAP_SEQ,
        s=dot_size(res_top["specificity_rank"].clip(lower=1e-4)),
        edgecolors="k",
        linewidths=0.3,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([REFERENCE, TEST])
    ax.set_xlim(-0.6, 1.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_ylim(-0.7, len(order) - 0.3)
    fig.colorbar(sca, ax=ax, shrink=0.4, label="lr_means")
    for v in (1, 2, 3):
        ax.scatter([], [], s=dot_size(10.0**-v), c="grey", label=f"{10**-v:g}")
    ax.legend(
        title="specificity_rank",
        bbox_to_anchor=(1.02, 0.02),
        loc="lower left",
        frameon=False,
        fontsize=7,
        title_fontsize=8,
    )
    ax.set_title("Top shifted interactions by condition")
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    savefig(fig, "A4_dotplot_by_condition")


def part_a(adata, groups) -> pd.DataFrame:
    """Run Part A; returns the combined per-condition LIANA table."""
    print("\n[Part A] Running LIANA rank_aggregate per condition ...")
    lr_genes = genes_in_resource(li.rs.select_resource(RESOURCE), adata.var_names)
    print(f"  {len(lr_genes)} of {adata.n_vars} genes are in the {RESOURCE} resource")

    per_condition = []
    for cond in CONDITIONS:
        adata_c = run_rank_aggregate(adata, cond, lr_genes)
        res_c = adata_c.uns["liana_res"].copy()
        res_c.to_csv(out_path(f"A0_rank_aggregate_{cond}", "csv"), index=False)
        plot_condition_results(adata_c, res_c, cond, groups)
        res_c.insert(0, CONDITION_KEY, cond)
        per_condition.append(res_c)
        del adata_c
        gc.collect()

    res = pd.concat(per_condition, ignore_index=True)
    res.to_csv(out_path("A_liana_per_condition", "csv"), index=False)
    res["sig"] = (res["magnitude_rank"] <= SIG_RANK) & (
        res["specificity_rank"] <= SIG_RANK
    )

    plot_a1_significant_counts(res, groups)
    wide = magnitude_shift_table(res)
    top = pd.concat([wide.head(TOP_N // 2), wide.tail(TOP_N // 2)]).drop_duplicates()
    plot_a2_shift_barplot(top)
    plot_a3_shift_scatter(wide, top)
    plot_a4_shift_dotplot(res, top)

    # A5: circle plots per condition (overview + one page per cell type)
    for cond in CONDITIONS:
        sig_c = res[(res[CONDITION_KEY] == cond) & res["sig"]].copy()
        circle_plots(
            adata, sig_c, f"A5_circle_{cond}", f"{cond}: significant L-R interactions", groups
        )  # fmt: skip
    return res


# =============================================================================
# PART B - Pseudobulk DESeq2 (TEST vs REFERENCE) mapped to L-R pairs
# =============================================================================
def build_pseudobulk(adata) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum raw counts per ROI x level_2; keep samples with enough cells."""
    obs = adata.obs[[SAMPLE_KEY, GROUP_KEY, CONDITION_KEY]].copy()
    obs["pb_id"] = obs[SAMPLE_KEY].astype(str) + "__" + obs[GROUP_KEY].astype(str)
    codes, uniques = pd.factorize(obs["pb_id"])
    indicator = sparse.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))),
        shape=(len(uniques), len(codes)),
    )
    pb_counts = pd.DataFrame(
        np.asarray((indicator @ adata.layers["counts"]).todense()).round().astype(int),
        index=uniques,
        columns=adata.var_names,
    )
    pb_meta = obs.drop_duplicates("pb_id").set_index("pb_id").loc[uniques]
    pb_meta["n_cells"] = np.bincount(codes)
    pb_meta = pb_meta[pb_meta["n_cells"] >= PB_MIN_CELLS]
    pb_meta.to_csv(out_path("B_pseudobulk_samples", "csv"))
    return pb_counts.loc[pb_meta.index], pb_meta


def run_deseq_per_cell_type(pb_counts, pb_meta, groups) -> pd.DataFrame:
    """PyDESeq2 TEST vs REFERENCE separately for each level_2 cell type."""
    results = []
    for cell_type in groups:
        meta = pb_meta[pb_meta[GROUP_KEY] == cell_type]
        reps = meta[CONDITION_KEY].value_counts().reindex(CONDITIONS, fill_value=0)
        if (reps < PB_MIN_SAMPLES_PER_COND).any():
            print(f"  skip {cell_type}: replicates {reps.to_dict()}")
            continue
        X = pb_counts.loc[meta.index]
        X = X.loc[:, (X >= GENE_MIN_COUNTS).sum(0) >= PB_MIN_SAMPLES_PER_COND]
        design = pd.DataFrame(
            {
                CONDITION_KEY: pd.Categorical(
                    meta[CONDITION_KEY].astype(str), categories=[REFERENCE, TEST]
                )
            },
            index=meta.index,
        )
        dds = DeseqDataSet(
            counts=X, metadata=design, design=f"~{CONDITION_KEY}", quiet=True
        )
        dds.deseq2()
        stats = DeseqStats(dds, contrast=[CONDITION_KEY, TEST, REFERENCE], quiet=True)
        stats.summary()
        df = stats.results_df.copy()
        df[GROUP_KEY] = cell_type
        results.append(df)
        print(
            f"  {cell_type}: {len(design)} pseudobulks, "
            f"{(df['padj'] < PADJ).sum()} DE genes (padj<{PADJ})"
        )
    if not results:
        raise RuntimeError(
            "No cell type had enough replicates per condition for DESeq2."
        )
    return pd.concat(results)


def map_de_to_lr(adata, dea_df) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map DE statistics onto L-R pairs.

    Returns (all interactions, coherently changed subset), where "coherent"
    means both partners move the same way and at least one is significant.
    """
    lr_dea = li.multi.df_to_lr(
        adata,
        dea_df=dea_df,
        resource_name=RESOURCE,
        expr_prop=EXPR_PROP,
        groupby=GROUP_KEY,
        stat_keys=["stat", "pvalue", "padj", "log2FoldChange"],
        use_raw=False,
        complex_col="stat",  # for complexes, keep the subunit with the lowest stat
        verbose=True,
        return_all_lrs=False,
    ).sort_values("interaction_stat", ascending=False)
    lr_dea.to_csv(
        out_path(f"B_LR_differential_{TEST}_vs_{REFERENCE}", "csv"), index=False
    )

    lr_dea["both_up"] = (lr_dea["ligand_stat"] > 0) & (lr_dea["receptor_stat"] > 0)
    lr_dea["both_down"] = (lr_dea["ligand_stat"] < 0) & (lr_dea["receptor_stat"] < 0)
    lr_dea["any_sig"] = (lr_dea["ligand_padj"] < PADJ) | (
        lr_dea["receptor_padj"] < PADJ
    )
    lr_dea["label"] = interaction_label(lr_dea)
    changed = lr_dea[lr_dea["any_sig"] & (lr_dea["both_up"] | lr_dea["both_down"])]
    changed.to_csv(out_path("B_LR_changed_filtered", "csv"), index=False)
    print(
        f"  {len(changed)} coherently changed L-R interactions "
        f"({changed['both_up'].sum()} up in {TEST}, {changed['both_down'].sum()} down)"
    )
    return lr_dea, changed


def select_top_changed(lr_dea, changed) -> pd.DataFrame:
    """Top up (both partners up in TEST) and top down interactions."""
    up = changed[changed["both_up"]].sort_values("interaction_stat", ascending=False)
    down = changed[changed["both_down"]].sort_values("interaction_stat")
    if len(up) + len(down) < 10:  # fall back to the unfiltered ranking
        up = lr_dea.sort_values("interaction_stat", ascending=False)
        down = lr_dea.sort_values("interaction_stat")
    return (
        pd.concat([up.head(TOP_N // 2), down.head(TOP_N // 2)])
        .drop_duplicates("label")
        .sort_values("interaction_stat")
    )


def plot_b1_heatmap(top_b) -> None:
    """Ligand / receptor / interaction statistics; * = padj < PADJ."""
    vals = top_b[["ligand_stat", "receptor_stat", "interaction_stat"]].values
    pvals = top_b[["ligand_padj", "receptor_padj"]].values
    lim = np.nanmax(np.abs(vals)) or 1
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(top_b) + 1.8))
    im = ax.imshow(vals, cmap=CMAP_DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i, j in zip(*np.nonzero(pvals < PADJ)):
        ax.text(j, i, "*", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Ligand\n(source)", "Receptor\n(target)", "Interaction"])
    ax.set_yticks(range(len(top_b)))
    ax.set_yticklabels(top_b["label"], fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.4, label=f"Wald stat ({TEST} vs {REFERENCE})")
    ax.set_title(f"Top differential L-R interactions\n* padj < {PADJ}")
    savefig(fig, "B1_heatmap_top_changed")


def plot_b2_barplot(top_b) -> None:
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(top_b) + 1.5))
    diverging_barh(ax, top_b["label"], top_b["interaction_stat"])
    ax.set_xlabel(
        f"Interaction stat (mean of ligand & receptor Wald stat, {TEST} vs {REFERENCE})"
    )
    ax.set_title(
        f"Differential L-R interactions\n(red = up in {TEST}, blue = up in {REFERENCE})"
    )
    ax.tick_params(axis="y", labelsize=8)
    savefig(fig, "B2_interaction_stat_barplot")


def plot_b3_ligand_vs_receptor(lr_dea, up, down) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        lr_dea["ligand_stat"], lr_dea["receptor_stat"], s=5, c="lightgrey", linewidths=0
    )
    for sub, color, label in (
        (up, UP_COLOR, f"up in {TEST}"),
        (down, DOWN_COLOR, f"up in {REFERENCE}"),
    ):
        ax.scatter(sub["ligand_stat"], sub["receptor_stat"], s=10, c=color, label=label)
    ax.axhline(0, c="k", lw=0.6)
    ax.axvline(0, c="k", lw=0.6)
    for _, r in pd.concat([up.head(6), down.tail(6)]).iterrows():
        ax.annotate(
            f"{r['ligand_complex']}→{r['receptor_complex']}",
            (r["ligand_stat"], r["receptor_stat"]),
            fontsize=6,
        )
    ax.set_xlabel("Ligand stat (in source)")
    ax.set_ylabel("Receptor stat (in target)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{TEST} vs {REFERENCE}")
    savefig(fig, "B3_ligand_vs_receptor_stat")


def plot_b4_source_target(up, down, groups) -> None:
    fig, axes = plt.subplots(1, 2, figsize=heatmap_figsize(len(groups), 2))
    for ax, (label, sub, cmap) in zip(
        axes, [(f"Up in {TEST}", up, "Reds"), (f"Up in {REFERENCE}", down, "Blues")]
    ):
        source_target_heatmap(
            fig, ax, count_matrix(sub, groups).values, groups,
            f"# L-R interactions {label}", cmap=cmap,
        )  # fmt: skip
    savefig(fig, "B4_changed_interactions_source_target")


def part_b(adata, groups) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Part B; returns (lr_dea, changed) for Part D."""
    print("\n[Part B] Pseudobulk differential expression ...")
    pb_counts, pb_meta = build_pseudobulk(adata)
    dea_df = run_deseq_per_cell_type(pb_counts, pb_meta, groups)
    dea_df.to_csv(out_path(f"B_pseudobulk_DEA_{TEST}_vs_{REFERENCE}", "csv"))
    lr_dea, changed = map_de_to_lr(adata, dea_df.dropna(subset=["stat"]))

    up = changed[changed["both_up"]]
    down = changed[changed["both_down"]]
    top_b = select_top_changed(lr_dea, changed)
    plot_b1_heatmap(top_b)
    plot_b2_barplot(top_b)
    plot_b3_ligand_vs_receptor(lr_dea, up, down)
    plot_b4_source_target(up, down, groups)
    circle_plots(
        adata, up.copy(), f"B5_circle_up_in_{TEST}",
        f"L-R interactions up in {TEST} vs {REFERENCE}", groups,
    )  # fmt: skip
    circle_plots(
        adata, down.copy(), f"B5_circle_up_in_{REFERENCE}",
        f"L-R interactions up in {REFERENCE} vs {TEST}", groups,
    )  # fmt: skip
    return lr_dea, changed


# =============================================================================
# PART C - CrossTalkeR: cell-type ranking on the communication graph
# =============================================================================
# Each condition's LIANA result (Part A) becomes a directed graph:
# nodes = level_2 cell types, edges = summed lr_means of the L-R pairs they
# exchange (only pairs with cellphone_pvals <= 0.05). The comparative graph
# "IPF_x_PM08" holds TEST - REFERENCE edge weights (red = higher in IPF,
# blue = higher in PM08). PageRank etc. are computed on every graph and
# differenced between conditions to find the cell types whose communication
# changes most. Fisher (number of interactions) and Mann-Whitney (strength)
# tests are run per source->target pair.
def crosstalker_cci(ctk_res, key, title, signed=False) -> None:
    """Wrapper around plot_cci that saves the figure.

    plot_cci sizes nodes in the order of graph.nodes(), so the PageRank values
    are re-ordered to match the graph before plotting. For the comparison a
    node that loses importance would get a negative size and vanish, so the
    absolute PageRank is used when `signed` is True.
    """
    edges = ctk_res["graphs"][key]
    graph = nx.from_pandas_edgelist(
        edges, source="source", target="target", create_using=nx.DiGraph()
    )
    pagerank = ctk_res["rankings"][key].set_index("nodes")["Pagerank"]
    if signed:
        pagerank = pagerank.abs()
    pagerank = pagerank.reindex(list(graph.nodes())).fillna(0)
    fig, _ = pyct_plots.plot.plot_cci(
        graph=edges,
        colors=ctk_res["colors"],
        plt_name=title,
        coords=ctk_res["coords"],
        emax=None,
        leg=False,
        low=0,
        high=0,
        ignore_alpha=False,
        log=False,
        efactor=2,
        vfactor=12,
        pg=pagerank,
        figsize=(7, 7),
        scale_factor=2.0,
        node_size=3.0,
        font_size=8,
        return_figure=True,
    )
    savefig(fig, f"{CTK_DIR}/C1_cci_{key}")


def plot_c2_pagerank(ctk_res) -> None:
    """PageRank in the comparison ranking = PageRank(TEST) - PageRank(REFERENCE).

    Influencer/Listener are computed on the difference graph itself, not as a
    difference between conditions, so they stay in the CSV but are not plotted.
    """
    pagerank = ctk_res["rankings"][CMP].set_index("nodes")["Pagerank"].sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(pagerank) + 1.8))
    diverging_barh(ax, pagerank.index, pagerank)
    ax.set_xlabel(f"Δ PageRank ({TEST} − {REFERENCE})")
    ax.set_ylabel(GROUP_KEY)
    ax.set_title(
        f"CrossTalkeR: change in cell-type importance\n"
        f"red = more central in {TEST}, blue = more central in {REFERENCE}"
    )
    savefig(fig, f"{CTK_DIR}/C2_pagerank_{CMP}")


def plot_c3_pair_heatmap(stats, value, title, name, groups) -> None:
    """Per source -> target value (e.g. log odds); * = p_value < 0.05."""
    d = stats.copy()
    d[["source", "target"]] = d["cellpair"].str.split("@", expand=True)
    vals = d.pivot(index="source", columns="target", values=value).reindex(
        index=groups, columns=groups
    )
    pvals = d.pivot(index="source", columns="target", values="p_value").reindex(
        index=groups, columns=groups
    )
    lim = np.nanmax(np.abs(vals.values)) or 1
    fig, ax = plt.subplots(
        figsize=(max(5, 0.5 * len(groups) + 2.5), max(4.5, 0.5 * len(groups) + 1.5))
    )
    source_target_heatmap(
        fig, ax, vals.values, groups, f"{title}\n* p < 0.05",
        cmap=CMAP_DIV, vmin=-lim, vmax=lim, cbar_label=value, stars=pvals.values < 0.05,
    )  # fmt: skip
    savefig(fig, f"{CTK_DIR}/{name}")


def part_c(res, adata, groups) -> None:
    print("\n[Part C] pyCrossTalkeR ...")
    lr_in = res.copy()
    lr_in["ligand"] = lr_in["ligand_complex"]  # from_liana expects a 'ligand' column
    lr_in[CONDITION_KEY] = lr_in[CONDITION_KEY].astype(str)
    ctk = ad.AnnData(uns={"liana_res": lr_in})
    ctk = pyct.tl.from_liana(
        ctk,
        liana_key="liana_res",
        score_key="lr_means",
        pval_key="cellphone_pvals",
        pval_filter=True,
        condition_key=CONDITION_KEY,
    )

    ctk_dir = OUTDIR / CTK_DIR
    ctk_dir.mkdir(parents=True, exist_ok=True)
    ctk = pyct.tl.analise_LR(
        ctk,
        out_path=str(ctk_dir),
        colors=dict(zip(groups, adata.uns[COLORS_KEY])),  # same colors as other plots
        comparison=[(TEST, REFERENCE)],  # (experiment, control)
        filename="crosstalker",
        save=False,
    )
    ctk_res = ctk.uns["pycrosstalker"]["results"]

    for key, ranking in ctk_res["rankings"].items():
        ranking.to_csv(ctk_dir / f"C_ranking_{key}.csv", index=False)
    ctk_res["stats"][CMP].to_csv(ctk_dir / f"C_fisher_{CMP}.csv", index=False)
    ctk_res["stats"][f"{CMP}:MannU"].to_csv(
        ctk_dir / f"C_mannwhitney_{CMP}.csv", index=False
    )
    ctk_res["graphs"][CMP].to_csv(ctk_dir / f"C_graph_edges_{CMP}.csv", index=False)

    for cond in CONDITIONS:
        crosstalker_cci(ctk_res, cond, cond)
    crosstalker_cci(
        ctk_res,
        CMP,
        f"Comparative ({TEST} vs {REFERENCE})\n"
        f"red = higher in {TEST}, blue = higher in {REFERENCE}",
        signed=True,
    )
    plot_c2_pagerank(ctk_res)
    plot_c3_pair_heatmap(
        ctk_res["stats"][CMP], "lodds",
        f"Number of interactions: log odds {TEST} vs {REFERENCE} (Fisher)",
        f"C3_fisher_{CMP}", groups,
    )  # fmt: skip
    plot_c3_pair_heatmap(
        ctk_res["stats"][f"{CMP}:MannU"], "lfc",
        f"Interaction strength: log fold change {TEST} vs {REFERENCE} (Mann-Whitney)",
        f"C3_mannwhitney_{CMP}", groups,
    )  # fmt: skip


# =============================================================================
# PART D - Spatial maps of the top differential L-R pairs
# =============================================================================
# For each top pair from Part B (both partners changed in the same direction),
# LIANA's bivariate metric scores, for every cell, how strongly the ligand and
# the receptor are co-expressed within its spatial neighborhood (Gaussian
# kernel, SPATIAL_BANDWIDTH um). Neighborhoods are built per ROI, because each
# ROI has its own coordinate frame. The local score (cosine similarity) is
# mapped onto the tissue, and summarized per ROI so the two conditions can be
# compared with ROIs as replicates.
@dataclass
class SpatialResults:
    local_scores: dict[str, pd.DataFrame]  # ROI -> cells x pairs
    coords: dict[str, np.ndarray]  # ROI -> (cells, 2)
    celltypes: dict[str, np.ndarray]  # ROI -> level_2 label per cell
    summary: pd.DataFrame  # ROI x pair: Moran's I + mean local score


def pretty(pair: str) -> str:
    return pair.replace("^", " → ")


def select_spatial_pairs(lr_dea, changed) -> pd.DataFrame:
    """Top up and top down pairs (unique ligand-receptor), saved to CSV."""

    def top_pairs(df, ascending, n):
        d = df.sort_values("interaction_stat", ascending=ascending)
        return d.drop_duplicates(["ligand_complex", "receptor_complex"]).head(n)

    pool = changed if len(changed) else lr_dea
    n_half = max(1, SPATIAL_TOP_PAIRS // 2)
    sel = pd.concat(
        [
            top_pairs(pool[pool["interaction_stat"] > 0], False, n_half),
            top_pairs(pool[pool["interaction_stat"] < 0], True, n_half),
        ]
    )
    sel["direction"] = np.where(
        sel["interaction_stat"] > 0, f"up in {TEST}", f"up in {REFERENCE}"
    )
    sel["pair"] = sel["ligand_complex"] + "^" + sel["receptor_complex"]
    sel[
        ["pair", "direction", "source", "target", "interaction_stat",
         "ligand_padj", "receptor_padj"]
    ].to_csv(out_path(f"{SPATIAL_DIR}/D_selected_pairs", "csv"), index=False)  # fmt: skip
    print("  Pairs mapped:", ", ".join(sel["pair"]))
    return sel


def score_rois(adata, sel, roi_meta, roi_order) -> SpatialResults:
    """Run LIANA bivariate (local cosine + global Moran's I) in each ROI."""
    resource = sel[["ligand_complex", "receptor_complex"]].rename(
        columns={"ligand_complex": "ligand", "receptor_complex": "receptor"}
    )
    genes = genes_in_resource(resource, adata.var_names)

    local_scores, coords, celltypes, rows = {}, {}, {}, []
    for roi in roi_order:
        a = adata[adata.obs[SAMPLE_KEY] == roi, genes].copy()
        if a.n_obs < SPATIAL_MIN_CELLS:
            print(f"  {roi}: only {a.n_obs} cells, skipped")
            continue
        li.ut.spatial_neighbors(
            a,
            spatial_key=SPATIAL_KEY,
            bandwidth=SPATIAL_BANDWIDTH,
            cutoff=SPATIAL_CUTOFF,
            kernel="gaussian",
            set_diag=True,
        )
        try:
            lrd = li.mt.bivariate(
                a,
                resource=resource,
                local_name="cosine",
                global_name="morans",
                n_perms=None,
                mask_negatives=False,
                add_categories=False,
                nz_prop=SPATIAL_NZ_PROP,
                use_raw=False,
                verbose=False,
            )
        except ValueError as e:  # no pair passes nz_prop in this ROI
            print(f"  {roi}: {e}")
            continue
        X = lrd.X.toarray() if sparse.issparse(lrd.X) else np.asarray(lrd.X)
        local_scores[roi] = pd.DataFrame(X, index=lrd.obs_names, columns=lrd.var_names)
        coords[roi] = np.asarray(a.obsm[SPATIAL_KEY])[:, :2]
        celltypes[roi] = a.obs[GROUP_KEY].astype(str).to_numpy()
        for pair, row in lrd.var.iterrows():
            rows.append(
                {
                    SAMPLE_KEY: roi,
                    CONDITION_KEY: roi_meta[roi],
                    "pair": pair,
                    "morans_I": row["morans"],
                    "mean_local_score": local_scores[roi][pair].mean(),
                }
            )
        print(f"  {roi}: {a.n_obs} cells, {lrd.n_vars} pairs scored")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path(f"{SPATIAL_DIR}/D_per_ROI_summary", "csv"), index=False)
    return SpatialResults(local_scores, coords, celltypes, summary)


def save_celltype_scores(sp: SpatialResults, roi_meta) -> None:
    """Mean local score per level_2 cell type and condition (where?)."""
    frames = [
        df.assign(**{GROUP_KEY: sp.celltypes[roi], CONDITION_KEY: roi_meta[roi]})
        for roi, df in sp.local_scores.items()
    ]
    if frames:
        pd.concat(frames).groupby([CONDITION_KEY, GROUP_KEY]).mean().to_csv(
            out_path(f"{SPATIAL_DIR}/D_mean_local_score_by_celltype", "csv")
        )


def plot_d1_tissue_maps(sp: SpatialResults, sel, pairs, rois, roi_meta) -> None:
    """One page per pair, all ROIs.

    The local cosine score is non-negative (0 = no co-expression, 1 = maximal),
    so a sequential map is correct; a diverging map would imply a meaningful
    midpoint / negative values that don't exist.
    """
    ncol = min(4, max(1, len(rois)))
    nrow = int(np.ceil(len(rois) / ncol)) if rois else 0
    info_by_pair = sel.set_index("pair")
    with PdfPages(out_path(f"{SPATIAL_DIR}/D1_spatial_maps_all_pairs", "pdf")) as pdf:
        for pair in pairs:
            all_vals = np.concatenate(
                [
                    sp.local_scores[r][pair].values
                    for r in rois
                    if pair in sp.local_scores[r]
                ]
            )
            vmax = np.nanpercentile(all_vals, 99) or 1
            fig, axes = plt.subplots(
                nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False
            )
            for ax in axes.flat:
                ax.axis("off")
            sca = None
            for ax, roi in zip(axes.flat, rois):
                xy = sp.coords[roi]
                if pair in sp.local_scores[roi]:
                    v = sp.local_scores[roi][pair].values
                    o = np.argsort(v)  # draw hotspots on top
                    sca = ax.scatter(
                        xy[o, 0], xy[o, 1], c=v[o], s=SPATIAL_POINT_SIZE,
                        cmap=CMAP_SEQ, vmin=0, vmax=vmax, linewidths=0, rasterized=True,
                    )  # fmt: skip
                else:
                    ax.scatter(
                        xy[:, 0], xy[:, 1], c="lightgrey", s=SPATIAL_POINT_SIZE,
                        linewidths=0, rasterized=True,
                    )  # fmt: skip
                    ax.text(
                        0.5, 0.02, "not expressed", transform=ax.transAxes,
                        ha="center", fontsize=7,
                    )  # fmt: skip
                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.set_title(
                    f"{roi} ({roi_meta[roi]})",
                    fontsize=9,
                    color=COND_COLORS.get(roi_meta[roi], "k"),
                )
            info = info_by_pair.loc[pair]
            fig.suptitle(
                f"{pretty(pair)}  ({info['direction']}; "
                f"{info['source']} → {info['target']})\n"
                f"local L-R co-expression (cosine, bandwidth {SPATIAL_BANDWIDTH} µm)",
                fontsize=11,
            )
            fig.colorbar(sca, ax=axes.ravel().tolist(), shrink=0.5, label="local score")
            fig.savefig(
                out_path(f"{SPATIAL_DIR}/D1_spatial_{safe_name(pair)}", "png"), dpi=300
            )
            pdf.savefig(fig)
            plt.close(fig)


def plot_d2_roi_comparison(summary, pairs) -> None:
    """Mean local score per ROI, REFERENCE vs TEST (ROIs as replicates)."""
    ncol = min(5, len(pairs))
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(2.6 * ncol, 3.2 * nrow), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    rng = np.random.default_rng(0)
    for ax, pair in zip(axes.flat, pairs):
        ax.axis("on")
        d = summary[summary["pair"] == pair]
        values_by_cond = []
        for i, cond in enumerate([REFERENCE, TEST]):
            y = d.loc[d[CONDITION_KEY] == cond, "mean_local_score"].values
            values_by_cond.append(y)
            ax.scatter(
                i + rng.uniform(-0.12, 0.12, len(y)), y, s=25,
                color=COND_COLORS[cond], edgecolors="k", linewidths=0.3,
            )  # fmt: skip
            if len(y):
                ax.hlines(np.median(y), i - 0.25, i + 0.25, color="k", lw=1.2)
        p = (
            mannwhitneyu(*values_by_cond).pvalue
            if all(len(v) >= 2 for v in values_by_cond)
            else np.nan
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([REFERENCE, TEST])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{pretty(pair)}\np = {p:.3g}", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("mean local score per ROI")
    fig.suptitle("Spatial L-R co-expression per ROI (Mann-Whitney, ROIs as replicates)")
    fig.tight_layout()
    savefig(fig, f"{SPATIAL_DIR}/D2_per_ROI_local_score")


def plot_d3_morans(summary, pairs, rois, roi_meta) -> None:
    """Global Moran's I (spatial clustering) per pair x ROI."""
    m = (
        summary.pivot(index="pair", columns=SAMPLE_KEY, values="morans_I")
        .reindex(index=pairs, columns=rois)
        .astype(float)
    )
    fig, ax = plt.subplots(figsize=(0.5 * len(rois) + 4, 0.4 * len(pairs) + 2))
    lim = np.nanmax(np.abs(m.values)) or 1
    im = ax.imshow(m.values, cmap=CMAP_DIV, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels(rois, rotation=90, fontsize=8)
    for tick, roi in zip(ax.get_xticklabels(), rois):
        tick.set_color(COND_COLORS.get(roi_meta[roi], "k"))
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([pretty(p) for p in pairs], fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Moran's I (bivariate)")
    ax.set_title(
        f"Global spatial co-localization of ligand & receptor\n"
        f"(ROI labels: red = {TEST}, blue = {REFERENCE})"
    )
    savefig(fig, f"{SPATIAL_DIR}/D3_morans_I_heatmap")


def part_d(adata, lr_dea, changed) -> None:
    print("\n[Part D] Spatial L-R maps ...")
    sel = select_spatial_pairs(lr_dea, changed)

    roi_meta = (
        adata.obs[[SAMPLE_KEY, CONDITION_KEY]]
        .drop_duplicates()
        .set_index(SAMPLE_KEY)[CONDITION_KEY]
        .astype(str)
    )
    roi_order = (  # TEST ROIs first
        roi_meta.sort_values(key=lambda c: c.map({TEST: 0, REFERENCE: 1}))
        .index.astype(str)
        .tolist()
    )

    sp = score_rois(adata, sel, roi_meta, roi_order)
    save_celltype_scores(sp, roi_meta)

    scored = set(sp.summary["pair"]) if not sp.summary.empty else set()
    pairs = [p for p in sel["pair"] if p in scored]
    rois = [r for r in roi_order if r in sp.local_scores]

    plot_d1_tissue_maps(sp, sel, pairs, rois, roi_meta)
    if pairs:
        plot_d2_roi_comparison(sp.summary, pairs)
        plot_d3_morans(sp.summary, pairs, rois, roi_meta)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

    adata = load_adata()
    print_summary(adata)
    normalize(adata)
    groups = sorted(adata.obs[GROUP_KEY].cat.categories)

    res = part_a(adata, groups)
    lr_dea, changed = part_b(adata, groups)
    part_c(res, adata, groups)
    part_d(adata, lr_dea, changed)

    print(f"\nDone. Results in: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
