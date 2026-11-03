"""Cell-cell communication (ligand-receptor) analysis: IPF vs PM08.

Two complementary analyses, both grouped by adata.obs["level_2"]:

  Part A - Per-condition LIANA consensus (rank_aggregate run separately for
           IPF and PM08). Answers: which interactions are strong/specific in
           each disease, and how does their magnitude (lr_means) shift?
           Descriptive; does not account for sample-to-sample variation.

  Part B - Sample-aware differential analysis. Pseudobulk (ROI x level_2),
           PyDESeq2 IPF vs PM08 (PM08 = reference) per cell type, then LIANA
           maps the DE statistics onto ligand-receptor pairs (li.multi.df_to_lr).
           This is the approach to use for statistical claims.

  Part C - CrossTalkeR: cell types ranked by PageRank on the communication
           graph, compared between conditions (+ Fisher / Mann-Whitney tests).

  Part D - Spatial: local co-expression (LIANA bivariate) of the top
           differential L-R pairs from Part B, mapped onto each ROI.

Requirements (tested with liana 1.10, pydeseq2 0.5.4, scanpy 1.11,
pycrosstalker 2.1.8):  see setup_venv_RL.sh

Outputs go to OUTDIR = <INPUT_DIR>/ccc/IPFvPM08 (CSV tables + PDF/PNG figures).
"""

import gc
import os
import warnings
from pathlib import Path

import anndata as ad
import liana as li
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import zarr
from liana.mt import rank_aggregate
from scanpy.plotting import palettes as _pal
from scipy import sparse

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIG - edit these
# =============================================================================
INPUT_DIR = Path("/rds/general/user/sep22/ephemeral/")
INPUT_ZARR = INPUT_DIR / "adata_with_metadata.zarr"
OUTDIR = INPUT_DIR / "output" / "ccc" / "IPFvPM08"

CONDITION_KEY = "condition"
CONDITIONS = ["IPF", "PM08"]
REFERENCE = "PM08"  # baseline: red = higher in IPF, blue = lower in IPF
TEST = "IPF"

GROUP_KEY = "level_2"  # cell types used as sources/targets
SAMPLE_KEY = "ROI"  # replicate unit for pseudobulk (Part B)
DROP_ROIS = ["PM08_159"]
# Cell types removed before analysis
DROP_CELL_TYPES = [
    "Alveolar fibroblasts (collagen high)",  # only present in one donor
]

# Where raw counts live. None = adata.X already holds raw counts.
# The script checks this and warns if the values don't look like counts.
COUNTS_LAYER = "counts"

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
SPATIAL_BANDWIDTH = 100  # um; Gaussian kernel width for neighbourhoods
SPATIAL_CUTOFF = 0.1  # drop neighbour weights below this
SPATIAL_TOP_PAIRS = 10  # top differential L-R pairs mapped (half up, half down)
SPATIAL_NZ_PROP = 0.05  # min fraction of cells expressing ligand & receptor per ROI
SPATIAL_POINT_SIZE = 0.5  # scatter point size on tissue maps

OUTDIR.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = OUTDIR
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

# Sequential colour map for non-negative values (counts, lr_means, local scores).
# Signed values (differences, stats) use the diverging "RdBu_r".
cmap = sns.color_palette("Blues", as_cmap=True)


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


_orig_from_adj = nx.convert_matrix.from_pandas_adjacency


def _square_adjacency(df, *args, **kwargs):
    labels = df.index.union(df.columns)
    df = df.reindex(index=labels, columns=labels, fill_value=0)
    return _orig_from_adj(df, *args, **kwargs)


nx.convert_matrix.from_pandas_adjacency = _square_adjacency


def circle_plots(adata, df, name, title, cell_types):
    """Circle plots of interaction counts between level_2 groups.

    Page 1 = all cell types; then one page per cell type showing its
    outgoing (as sender) and incoming (as receiver) interactions.
    Saved as one multi-page PDF plus a PNG of the overview.
    Edge width = number of interactions; widths are scaled within each plot,
    so compare across plots using the counts heatmaps, not edge widths.
    """
    from matplotlib.backends.backend_pdf import PdfPages

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
    with PdfPages(os.path.join(OUTDIR, f"{name}.pdf")) as pdf:
        ax = li.pl.circle_plot(adata, **kw)
        ax.set_title(f"{title}\nall {GROUP_KEY} groups (n = {len(df)})")
        fig = ax.get_figure()
        fig.savefig(os.path.join(OUTDIR, f"{name}_overview.png"), dpi=300)
        pdf.savefig(fig)
        plt.close(fig)
        for ct in cell_types:
            for role, sub, lab in [
                ("source", df[df["source"] == ct], "outgoing"),
                ("target", df[df["target"] == ct], "incoming"),
            ]:
                if sub.empty:
                    continue
                ax = li.pl.circle_plot(
                    adata, **{**kw, "liana_res": sub}, **{f"{role}_labels": [ct]}
                )
                ax.set_title(f"{title}\n{ct}: {lab} (n = {len(sub)})")
                fig = ax.get_figure()
                pdf.savefig(fig)
                plt.close(fig)


def looks_like_counts(X):
    vals = X[:1000].toarray() if sparse.issparse(X) else np.asarray(X[:1000])
    return np.all(vals >= 0) and np.allclose(vals, np.round(vals))


def count_matrix(df, cond):
    sub = df[(df[CONDITION_KEY] == cond) & df["sig"]]
    m = pd.crosstab(sub["source"], sub["target"])
    return m


def crosstalker_cci(key, title, signed=False, **kw):
    """Wrapper around ctpl.plot.plot_cci that saves the figure.

    plot_cci sizes nodes in the order of graph.nodes(), so the PageRank values
    are re-ordered to match the graph before plotting.
    """
    import networkx as nx

    edges = ctk_res["graphs"][key]
    g = nx.from_pandas_edgelist(
        edges, source="source", target="target", create_using=nx.DiGraph()
    )
    pr = ctk_res["rankings"][key].set_index("nodes")["Pagerank"]
    # node size = |PageRank|; for the comparison a node that loses importance
    # would otherwise get a negative size and vanish
    pg = pr.abs() if signed else pr
    pg = pg.reindex(list(g.nodes())).fillna(0)
    args = dict(
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
        pg=pg,
        figsize=(7, 7),
        scale_factor=2.0,
        node_size=3.0,
        font_size=8,
        return_figure=True,
    )
    args.update(kw)
    fig, ax = ctpl.plot.plot_cci(**args)
    savefig(fig, os.path.join("C_crosstalker", f"C1_cci_{key}"))


def pair_heatmap(df, value, pcol, title, name):
    d = df.copy()
    d[["source", "target"]] = d["cellpair"].str.split("@", expand=True)
    vals = d.pivot(index="source", columns="target", values=value).reindex(
        index=groups, columns=groups
    )
    pv = d.pivot(index="source", columns="target", values=pcol).reindex(
        index=groups, columns=groups
    )
    lim = np.nanmax(np.abs(vals.values)) or 1
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * n + 2.5), max(4.5, 0.5 * n + 1.5)))
    im = ax.imshow(vals.values, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(n):
        for j in range(n):
            p_ij = pv.values[i, j]
            if np.isfinite(p_ij) and p_ij < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(n))
    ax.set_xticklabels(groups, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(groups)
    ax.set_xlabel("Target (receptor)")
    ax.set_ylabel("Source (ligand)")
    fig.colorbar(im, ax=ax, shrink=0.7, label=value)
    ax.set_title(title + "\n* p < 0.05")
    savefig(fig, os.path.join("C_crosstalker", name))


def pretty(pair):
    return pair.replace("^", ARROW)


# ---- choose pairs: top up and top down (unique ligand-receptor) ------------
def top_pairs(df, ascending, n):
    d = df.sort_values("interaction_stat", ascending=ascending)
    d = d.drop_duplicates(["ligand_complex", "receptor_complex"])
    return d.head(n)


# =============================================================================
# 1. Load and subset
# =============================================================================
# Memory-efficient load: the full object has dense X + dense counts (~25 GB
# each), so instead of ad.read_zarr() we read only what this script needs
# (obs columns, var names, spatial coords, level_2 colours, raw counts), only
# for the cells we keep, and convert counts to sparse float32 in row chunks.

try:
    from anndata.io import read_elem, sparse_dataset
except ImportError:  # older anndata
    from anndata.experimental import read_elem, sparse_dataset

LOAD_CHUNK = 20000  # rows read at a time when the matrix is stored dense

g = zarr.open(str(INPUT_ZARR), mode="r")
obs_all = read_elem(g["obs"])[[CONDITION_KEY, SAMPLE_KEY, GROUP_KEY]]
var = read_elem(g["var"])[[]]  # gene names only

# Remember the existing level_2 colours by name so they survive subsetting
_orig_colors = {}
if "uns" in g and f"{GROUP_KEY}_colors" in g["uns"]:
    _cols = list(read_elem(g["uns"][f"{GROUP_KEY}_colors"]))
    _lab = obs_all[GROUP_KEY]
    if isinstance(_lab.dtype, pd.CategoricalDtype) and len(_cols) == len(
        _lab.cat.categories
    ):
        _orig_colors = dict(zip(_lab.cat.categories.astype(str), _cols))

# Which cells to keep (decided from obs before touching the matrix)
lab = obs_all[GROUP_KEY]
unlabelled = lab.isna() | lab.astype(str).str.lower().isin(["nan", "none", ""])
keep = (
    obs_all[CONDITION_KEY].isin(CONDITIONS)  # IPF and PM08 only
    & ~obs_all[SAMPLE_KEY].isin(DROP_ROIS)  # exclude PM08_159
    & ~lab.isin(DROP_CELL_TYPES)  # drop listed cell types
).values
n_unlab = int((keep & unlabelled.values).sum())
keep &= ~unlabelled.values  # cells without a level_2 label break the plots
print(f"Removing {n_unlab} cells without a {GROUP_KEY} label")
print(f"Keeping {keep.sum()} of {len(keep)} cells")
idx = np.flatnonzero(keep)

# Raw counts: from layers[COUNTS_LAYER] if present, else X
if COUNTS_LAYER is not None and "layers" in g and COUNTS_LAYER in g["layers"]:
    src = g["layers"][COUNTS_LAYER]
else:
    src = g["X"]
enc = dict(src.attrs).get("encoding-type", "array")
if enc == "csr_matrix":
    print("Counts stored sparse (CSR); reading selected rows")
    counts = sparse_dataset(src)[idx].tocsr().astype(np.float32)
elif enc == "csc_matrix":
    print("Counts stored sparse (CSC); reading then subsetting")
    counts = read_elem(src).tocsr()[idx].astype(np.float32)
else:
    print(f"Counts stored DENSE {src.shape}; converting to sparse in chunks")
    blocks = []
    for start in range(0, src.shape[0], LOAD_CHUNK):
        stop = min(start + LOAD_CHUNK, src.shape[0])
        rows = keep[start:stop]
        if rows.any():
            blk = np.asarray(src[start:stop], dtype=np.float32)[rows]
            blocks.append(sparse.csr_matrix(blk))
            del blk
    counts = sparse.vstack(blocks, format="csr")
    del blocks
gc.collect()

spatial = np.asarray(read_elem(g["obsm"][SPATIAL_KEY]))[idx]
adata = ad.AnnData(
    X=counts,
    obs=obs_all.iloc[idx].copy(),
    var=var,
    obsm={SPATIAL_KEY: spatial},
    layers={"counts": counts},
)
COUNTS_LAYER = "counts"
del obs_all, spatial
gc.collect()

# Convert relevant columns to categorical and reorder
for key in (CONDITION_KEY, GROUP_KEY, SAMPLE_KEY):
    adata.obs[key] = adata.obs[key].astype(str).astype("category")
adata.obs[CONDITION_KEY] = adata.obs[CONDITION_KEY].cat.reorder_categories(
    [REFERENCE, TEST]
)

# Re-align level_2 colours with the remaining categories (new ones get defaults)
# (uses scanpy's public palettes, so it works across scanpy versions)


_cats = adata.obs[GROUP_KEY].cat.categories
_n = len(_cats)
_default = (
    _pal.default_20 if _n <= 20 else _pal.default_28 if _n <= 28 else _pal.default_102
)
_default = [_default[i % len(_default)] for i in range(_n)]
adata.uns[f"{GROUP_KEY}_colors"] = [
    _orig_colors.get(c, d) for c, d in zip(_cats, _default)
]

print(adata)
print("\nCells per condition x level_2:")
print(pd.crosstab(adata.obs[GROUP_KEY], adata.obs[CONDITION_KEY]))
print("\nCells per ROI x level_2:")
print(pd.crosstab(adata.obs[GROUP_KEY], adata.obs[CONDITION_KEY]))
print("\nROIs per condition:")
print(adata.obs.groupby(CONDITION_KEY, observed=True)[SAMPLE_KEY].nunique())


# =============================================================================
# 2. Counts layer + log-normalized X
# =============================================================================

if COUNTS_LAYER is not None and COUNTS_LAYER in adata.layers:
    counts = adata.layers[COUNTS_LAYER]
elif looks_like_counts(adata.X):
    counts = adata.X.copy()
else:
    raise ValueError(
        f"No raw counts found: layer '{COUNTS_LAYER}' is missing and adata.X "
        "is not integer counts. Point COUNTS_LAYER at your raw counts."
    )

if not looks_like_counts(counts):
    warnings.warn(
        "The counts layer doesn't look like integer counts. "
        "DESeq2 in Part B needs raw counts."
    )

adata.layers["counts"] = counts  # raw counts kept for Part B (no extra copy)
adata.X = counts.copy()  # normalised in place below
del counts
gc.collect()
sc.pp.normalize_total(adata, target_sum=1e4)
adata.uns.pop("log1p", None)  # stale flag from the saved object; X is raw counts here
sc.pp.log1p(adata)
adata.raw = None  # make sure LIANA uses X

# =============================================================================
# PART A - LIANA consensus per condition
# =============================================================================
# rank_aggregate runs CellPhoneDB, Connectome, log2FC, NATMI, SingleCellSignalR
# and CellChat-style scoring, then aggregates their ranks into
# magnitude_rank (strength) and specificity_rank (cell-type specificity).
# Run separately per condition so IPF and PM08 are not pooled.

print("\n[Part A] Running LIANA rank_aggregate per condition ...")
groups_all = adata.obs[GROUP_KEY].cat.categories.tolist()

# LIANA scales (densifies) the expression matrix; restricting to genes in the
# resource cuts memory a lot and doesn't change the results.
_rs = li.rs.select_resource(RESOURCE)
lr_genes = sorted(
    {g for x in pd.concat([_rs["ligand"], _rs["receptor"]]) for g in str(x).split("_")}
    & set(adata.var_names)
)
print(f"  {len(lr_genes)} of {adata.n_vars} genes are in the {RESOURCE} resource")

res_list = []
for c in CONDITIONS:
    adata_c = adata[adata.obs[CONDITION_KEY] == c, lr_genes].copy()
    adata_c.layers.clear()  # counts not needed here
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
    res_c = adata_c.uns["liana_res"].copy()
    res_c.to_csv(os.path.join(OUTDIR, f"A0_rank_aggregate_{c}.csv"), index=False)
    # Save adata object
    adata_c.write_zarr(OUTDIR / f"ccc_{c}.zarr")
    print(f" saved file to {OUTDIR}/ccc_{c}.zarr")
