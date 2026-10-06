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
import re
import warnings
from pathlib import Path

import anndata as ad
import liana as li
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine
import scanpy as sc
import seaborn as sns
from scanpy.plotting import palettes as _pal

warnings.filterwarnings("ignore", category=FutureWarning)


def count_matrix(df, cond):
    sub = df[(df[CONDITION_KEY] == cond) & df["sig"]]
    m = pd.crosstab(sub["source"], sub["target"])
    return m


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


# CONFIG
INPUT_DIR = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output"
)
INPUT_ZARR = INPUT_DIR / "AIRSCAPE/adata_final_object/adata_with_metadata.zarr"
OUTDIR = INPUT_DIR / "ccc" / "IPFvPM08"

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
TOP_N = 25  # number of interactions to show in dotplots (overview and per source)
SIG_RANK = 0.05  # threshold for "significant" interactions
DOTPLOT_PVAL = 0.05  # threshold for dotplots by source (A6)
DOTPLOT_TOP_N = 25  # max number of interactions to show in dotplots by source (A6)


OUTDIR.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = OUTDIR
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

# Sequential colour map for non-negative values (counts, lr_means, local scores).
# Signed values (differences, stats) use the diverging "RdBu_r".
cmap = sns.color_palette("Blues", as_cmap=True)
CMAP_NAME = "Blues"

# 1. Load and subset
adata = ad.read_zarr(INPUT_ZARR)

# Subset to IPF and PM08 samples
adata = adata[adata.obs[CONDITION_KEY].isin(CONDITIONS)]
# Exclude PM08_159
adata = adata[~adata.obs[SAMPLE_KEY].isin(DROP_ROIS)]
# Remove cell types listed in DROP_CELL_TYPES
adata = adata[~adata.obs[GROUP_KEY].isin(DROP_CELL_TYPES)]
# Remove cells without a cell-type label (NaN / "nan") - they break the plots
lab = adata.obs[GROUP_KEY]
unlabelled = lab.isna() | lab.astype(str).str.lower().isin(["nan", "none", ""])
print(f"Removing {int(unlabelled.sum())} cells without a {GROUP_KEY} label")
adata = adata[~unlabelled].copy()  # .copy() avoids view warnings

# Convert relevant columns to categorical and reorder
for key in (CONDITION_KEY, GROUP_KEY, SAMPLE_KEY):
    adata.obs[key] = adata.obs[key].astype(str).astype("category")
adata.obs[CONDITION_KEY] = adata.obs[CONDITION_KEY].cat.reorder_categories(
    [REFERENCE, TEST]
)

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
# adata.uns[f"{GROUP_KEY}_colors"] = [
#     _orig_colors.get(c, d) for c, d in zip(_cats, _default)
# ]

print(adata)
print("\nCells per condition x level_2:")
print(pd.crosstab(adata.obs[GROUP_KEY], adata.obs[CONDITION_KEY]))
print("\nROIs per condition:")
print(adata.obs.groupby(CONDITION_KEY, observed=True)[SAMPLE_KEY].nunique())

print("Running LIANA rank_aggregate per condition ...")
groups_all = adata.obs[GROUP_KEY].cat.categories.tolist()

# Restricting LIANA to genes in the resources
_rs = li.rs.select_resource(RESOURCE)
lr_genes = sorted(
    {g for x in pd.concat([_rs["ligand"], _rs["receptor"]]) for g in str(x).split("_")}
    & set(adata.var_names)
)
print(f"{len(lr_genes)} of {adata.n_vars} genes are in the {RESOURCE} resource")

print("Running rank_aggregate for each condition ...")
res_list = []
for c in CONDITIONS:
    adata_c = adata[adata.obs[CONDITION_KEY] == c, lr_genes].copy()
    adata_c.layers.clear()  # counts not needed here
    li.mt.rank_aggregate(
        adata_c,
        groupby=GROUP_KEY,
        resource_name=RESOURCE,
        expr_prop=EXPR_PROP,
        min_cells=MIN_CELLS_PER_GROUP,
        n_perms=N_PERMS,
        use_raw=False,  # log-normalized data is in X (adata.raw is None)
        verbose=False,  # True prints a 1000-line permutation progress bar
    )

    # Save results to CSV for each condition
    res_c = adata_c.uns["liana_res"].copy()
    res_c.to_csv(os.path.join(OUTDIR, f"A0_rank_aggregate_{c}.csv"), index=False)

    # LIANA dotplot of the top interactions for each condition (overview, all cell types)
    p = (
        li.pl.dotplot(
            adata=adata_c,
            colour="magnitude_rank",
            size="specificity_rank",
            inverse_colour=True,  # small ranks = strong -> bright
            inverse_size=True,  # small ranks = specific -> large
            # only cell types that appear in this condition's results
            source_labels=[g for g in groups_all if g in set(res_c["source"])],
            target_labels=[g for g in groups_all if g in set(res_c["target"])],
            top_n=TOP_N,  # show the top N interactions
            orderby="magnitude_rank",
            orderby_ascending=True,
            cmap=CMAP_NAME,
            figure_size=(max(10, 2 * len(groups_all) + 8), max(6, 0.3 * TOP_N + 2)),
        )
        + plotnine.ggtitle(f"{c}: top {TOP_N} interactions (rank_aggregate)")
        + plotnine.labs(
            colour="-log10(magnitude_rank)", size="-log10(specificity_rank)"
        )
    )
    p.save(
        os.path.join(OUTDIR, f"A0_rank_aggregate_dotplot_{c}.png"),
        dpi=300,
        verbose=False,
        limitsize=False,
    )
    p.save(
        os.path.join(OUTDIR, f"A0_rank_aggregate_dotplot_{c}.pdf"),
        verbose=False,
        limitsize=False,
    )

    # Circle plot of the whole network
    # edge width = number of interactions with magnitude_rank <= SIG_RANK
    adata_c.uns[f"{GROUP_KEY}_colors"] = adata.uns[f"{GROUP_KEY}_colors"]
    ax = li.pl.circle_plot(
        adata_c,
        groupby=GROUP_KEY,
        score_key="magnitude_rank",
        inverse_score=True,
        filter_fun=lambda x: x["magnitude_rank"] <= SIG_RANK,
        pivot_mode="counts",
        figure_size=(7, 7),
    )
    ax.set_title(f"{c}: # interactions (magnitude_rank ≤ {SIG_RANK})")
    fig = ax.get_figure()
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(OUTDIR, f"A0_rank_aggregate_circle_{c}.{ext}"), dpi=300
        )
    plt.close(fig)

    res_c.insert(0, CONDITION_KEY, c)
    res_list.append(res_c)
    gc.collect()

    # Create dotplot per sender cell type (source -> all targets)
    # e.g. A0_dotplots_by_source/IPF/Interstitial_macrophages.png
    # Same colour/size as the overview: magnitude_rank / specificity_rank.

    src_dir = os.path.join(OUTDIR, "A0_dotplots_by_source", c)
    os.makedirs(src_dir, exist_ok=True)
    pages = []
    for celltype in groups_all:
        # Get the subset of interactions where this cell type is the sender
        sub = res_c[res_c["source"] == celltype]
        if sub.empty:
            print(f"  A0 {c} / {celltype}: no interactions as sender, skipped")
            continue
        targets = [g for g in groups_all if g in set(sub["target"])]
        n_int = min(
            TOP_N,
            sub[["ligand_complex", "receptor_complex"]].drop_duplicates().shape[0],
        )
        p_ct = (
            li.pl.dotplot(
                liana_res=res_c,
                colour="magnitude_rank",
                size="specificity_rank",
                inverse_colour=True,
                inverse_size=True,
                source_labels=[celltype],
                target_labels=targets,
                top_n=TOP_N,
                orderby="magnitude_rank",
                orderby_ascending=True,
                cmap=CMAP_NAME,
                figure_size=(max(8, 0.45 * len(targets) + 4), max(5, 0.28 * n_int + 2)),
            )
            + plotnine.ggtitle(f"{c}: {celltype} \u2192 all {GROUP_KEY} (top {TOP_N})")
            + plotnine.labs(
                colour="-log10(magnitude_rank)", size="-log10(specificity_rank)"
            )
        )
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(celltype)).strip("_")
        p_ct.save(
            os.path.join(src_dir, f"{safe}.png"),
            dpi=300,
            verbose=False,
            limitsize=False,
        )
        pages.append(p_ct)
    if pages:
        plotnine.save_as_pdf_pages(
            pages,
            filename=os.path.join(OUTDIR, f"A0_dotplots_by_source_{c}.pdf"),
            verbose=False,
        )

    # Plot circle plot per source cell
    circ_dir = os.path.join(OUTDIR, "A0_circle_by_celltype", c)
    os.makedirs(circ_dir, exist_ok=True)
    ax = li.pl.circle_plot(
        adata_c,
        uns_key="liana_sig",
        groupby=GROUP_KEY,
        pivot_mode="counts",
        source_labels=celltype,  # None = whole network
        figure_size=(7, 7),
    )
    name = "all_cell_types" if celltype is None else celltype
    ax.set_title(f"{c}: {name} (magnitude_rank \u2264 {SIG_RANK})")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    ax.get_figure().savefig(
        os.path.join(circ_dir, f"{safe}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(ax.get_figure())

    # Clean up to save memory
    del adata_c
    gc.collect()


# ---- A1: number of significant interactions per source->target ------------
# Combined table (both conditions) used by the plots below
adata.uns["liana_res"] = pd.concat(res_list, ignore_index=True)
res = adata.uns["liana_res"].copy()
res.to_csv(os.path.join(OUTDIR, "A_liana_per_condition.csv"), index=False)

res["interaction"] = res["ligand_complex"] + " → " + res["receptor_complex"]
res["pair"] = res["source"].astype(str) + " → " + res["target"].astype(str)
res["sig"] = (res["magnitude_rank"] <= SIG_RANK) & (res["specificity_rank"] <= SIG_RANK)

groups = sorted(adata.obs[GROUP_KEY].cat.categories)
mats = {
    c: count_matrix(res, c).reindex(index=groups, columns=groups, fill_value=0)
    for c in CONDITIONS
}
diff = mats[TEST] - mats[REFERENCE]

n = len(groups)
fig, axes = plt.subplots(
    1, 3, figsize=(3 * max(5, 0.45 * n + 2), max(4.5, 0.45 * n + 1.5))
)
vmax = max(m.values.max() for m in mats.values()) or 1
for ax, c in zip(axes[:2], [REFERENCE, TEST]):
    im = ax.imshow(mats[c].values, cmap=CMAP_NAME, vmin=0, vmax=vmax)
    ax.set_title(f"{c}: # significant L-R")
    fig.colorbar(im, ax=ax, shrink=0.7)
lim = np.abs(diff.values).max() or 1
im = axes[2].imshow(diff.values, cmap="RdBu_r", vmin=-lim, vmax=lim)
axes[2].set_title(f"{TEST} − {REFERENCE}")
fig.colorbar(im, ax=axes[2], shrink=0.7)
for ax in axes:
    ax.set_xticks(range(n))
    ax.set_xticklabels(groups, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(groups)
    ax.set_xlabel("Target (receptor)")
    ax.set_ylabel("Source (ligand)")
savefig(fig, "A1_n_significant_interactions_heatmaps")

# ---- A2: magnitude shift (lr_means) for interactions sig in either condition -
wide = res.pivot_table(
    index=["source", "target", "ligand_complex", "receptor_complex"],
    columns=CONDITION_KEY,
    values="lr_means",
    observed=True,
)
sig_any = res.loc[
    res["sig"], ["source", "target", "ligand_complex", "receptor_complex"]
].drop_duplicates()
wide = wide.reset_index().merge(sig_any, how="inner")
wide[[REFERENCE, TEST]] = wide[[REFERENCE, TEST]].fillna(0)  # not detected -> 0
wide["delta_lr_means"] = wide[TEST] - wide[REFERENCE]
wide["log2FC_lr_means"] = np.log2((wide[TEST] + 1e-3) / (wide[REFERENCE] + 1e-3))
wide["label"] = (
    wide["ligand_complex"]
    + " → "
    + wide["receptor_complex"]
    + "  ("
    + wide["source"].astype(str)
    + " → "
    + wide["target"].astype(str)
    + ")"
)
wide = wide.sort_values("delta_lr_means")
wide.to_csv(os.path.join(OUTDIR, "A_lr_means_delta.csv"), index=False)

top = pd.concat([wide.head(TOP_N // 2), wide.tail(TOP_N // 2)]).drop_duplicates()
fig, ax = plt.subplots(figsize=(8, 0.28 * len(top) + 1.5))
colors = np.where(top["delta_lr_means"] > 0, "#c0392b", "#2c6fbb")
ax.barh(top["label"], top["delta_lr_means"], color=colors)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel(f"Δ mean L-R expression ({TEST} − {REFERENCE})")
ax.set_title(
    f"Largest L-R magnitude shifts\n(red = higher in {TEST}, blue = higher in {REFERENCE})"
)
ax.tick_params(axis="y", labelsize=8)
savefig(fig, "A2_top_delta_lr_means_barplot")

# ---- A3: REFERENCE vs TEST scatter of lr_means --------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(wide[REFERENCE], wide[TEST], s=6, c="grey", alpha=0.5, linewidths=0)
ax.scatter(top[REFERENCE], top[TEST], s=14, c=colors)
mx = max(wide[REFERENCE].max(), wide[TEST].max())
ax.plot([0, mx], [0, mx], "k--", lw=0.8)
for _, r in top.tail(8).iterrows():
    ax.annotate(
        r["ligand_complex"] + "→" + r["receptor_complex"],
        (r[REFERENCE], r[TEST]),
        fontsize=6,
    )
for _, r in top.head(8).iterrows():
    ax.annotate(
        r["ligand_complex"] + "→" + r["receptor_complex"],
        (r[REFERENCE], r[TEST]),
        fontsize=6,
    )
ax.set_xlabel(f"lr_means in {REFERENCE}")
ax.set_ylabel(f"lr_means in {TEST}")
ax.set_title("L-R magnitude per source→target pair")
savefig(fig, "A3_lr_means_scatter")

# ---- A4: side-by-side dotplot of the top shifted interactions -------------
# colour = lr_means (magnitude), size = -log10(specificity_rank)
key_cols = ["source", "target", "ligand_complex", "receptor_complex"]
res_top = res.merge(top[key_cols + ["label", "delta_lr_means"]], how="inner")
order = top.sort_values("delta_lr_means")["label"].tolist()
ypos = {l: i for i, l in enumerate(order)}
xpos = {REFERENCE: 0, TEST: 1}
fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(order) + 1.8))
sca = ax.scatter(
    res_top[CONDITION_KEY].astype(str).map(xpos),
    res_top["label"].map(ypos),
    c=res_top["lr_means"],
    cmap=CMAP_NAME,
    s=20 + 25 * -np.log10(res_top["specificity_rank"].clip(lower=1e-4)),
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
    ax.scatter([], [], s=20 + 25 * v, c="grey", label=f"{10**-v:g}")
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

# ---- A5: dotplot per source cell type -> all targets, per condition ---------
# colour = lr_means, size = -log10(cellphone_pvals); CellPhoneDB-style filter
# One multi-page PDF per condition (one page per source) + one PNG per source.

cell_list = adata.obs[GROUP_KEY].cat.categories.tolist()
dot_dir = os.path.join(OUTDIR, "A5_dotplots_by_source")
os.makedirs(dot_dir, exist_ok=True)

for c in CONDITIONS:
    res_c = adata.uns["liana_res"]
    res_c = res_c[res_c[CONDITION_KEY] == c]
    pages = []
    for celltype in cell_list:
        sub = res_c[
            (res_c["source"] == celltype) & (res_c["cellphone_pvals"] <= DOTPLOT_PVAL)
        ]
        if sub.empty:
            print(
                f"  A5 {c} / {celltype}: no interactions with p <= {DOTPLOT_PVAL}, skipped"
            )
            continue
        n_int = sub[["ligand_complex", "receptor_complex"]].drop_duplicates().shape[0]
        n_show = n_int if DOTPLOT_TOP_N is None else min(n_int, DOTPLOT_TOP_N)
        p = (
            li.pl.dotplot(
                liana_res=res_c,
                colour="lr_means",
                size="cellphone_pvals",
                inverse_size=True,  # small p-values -> large dots
                source_labels=[celltype],
                target_labels=[g for g in cell_list if g in set(res_c["target"])],
                filter_fun=lambda x: x["cellphone_pvals"] <= DOTPLOT_PVAL,
                top_n=DOTPLOT_TOP_N,
                cmap=CMAP_NAME,
                orderby="lr_means" if DOTPLOT_TOP_N else None,
                orderby_ascending=False if DOTPLOT_TOP_N else None,
                figure_size=(
                    max(8, 0.45 * len(cell_list) + 4),
                    max(5, 0.25 * n_show + 2),
                ),
            )
            + plotnine.ggtitle(
                f"{c}: {celltype} → all {GROUP_KEY} (cellphone_pvals ≤ {DOTPLOT_PVAL})"
            )
            + plotnine.labs(size="-log10(p)", colour="lr_means")
        )
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(celltype))
        p.save(
            os.path.join(dot_dir, f"{c}_{safe}.png"),
            dpi=300,
            verbose=False,
            limitsize=False,
        )
        pages.append(p)
    if pages:
        plotnine.save_as_pdf_pages(
            pages,
            filename=os.path.join(OUTDIR, f"A6_dotplots_by_source_{c}.pdf"),
            verbose=False,
        )
