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

import os
import re
import warnings
from pathlib import Path

import anndata as ad
import liana as li
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from liana.mt import rank_aggregate
from plotnine import ggtitle, labs
from scipy import sparse

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIG - edit these
# =============================================================================
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

cmap = sns.color_palette("Blues", as_cmap=True)


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


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


# =============================================================================
# 1. Load and subset
# =============================================================================
adata = ad.read_zarr(INPUT_ZARR)

# Subset to IPF and PM08 samples
adata = adata[adata.obs[CONDITION_KEY].isin(CONDITIONS)]
# Exclude PM08_159
adata = adata[~adata.obs[SAMPLE_KEY].isin(DROP_ROIS)]
# Remove cell types listed in DROP_CELL_TYPES
adata = adata[
    ~adata.obs[GROUP_KEY].isin(DROP_CELL_TYPES)
].copy()  # .copy() avoids view warnings

# Convert relevant columns to categorical and reorder
for key in (CONDITION_KEY, GROUP_KEY, SAMPLE_KEY):
    adata.obs[key] = adata.obs[key].astype(str).astype("category")
adata.obs[CONDITION_KEY] = adata.obs[CONDITION_KEY].cat.reorder_categories(
    [REFERENCE, TEST]
)

print(adata)
print("\nCells per condition x level_2:")
print(pd.crosstab(adata.obs[GROUP_KEY], adata.obs[CONDITION_KEY]))
print("\nROIs per condition:")
print(adata.obs.groupby(CONDITION_KEY, observed=True)[SAMPLE_KEY].nunique())


# =============================================================================
# 2. Counts layer + log-normalized X
# =============================================================================
def looks_like_counts(X):
    vals = X[:1000].toarray() if sparse.issparse(X) else np.asarray(X[:1000])
    return np.all(vals >= 0) and np.allclose(vals, np.round(vals))


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

adata.layers["counts"] = counts.copy()
adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
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
if f"{GROUP_KEY}_colors" not in adata.uns:  # same node colours in every plot
    sc.pl._utils._set_default_colors_for_categorical_obs(adata, GROUP_KEY)
res_list = []
for c in CONDITIONS:
    adata_c = adata[adata.obs[CONDITION_KEY] == c].copy()
    rank_aggregate(
        adata_c,
        groupby=GROUP_KEY,
        resource_name=RESOURCE,
        expr_prop=EXPR_PROP,
        min_cells=MIN_CELLS_PER_GROUP,
        n_perms=N_PERMS,
        use_raw=False,  # log-normalized data is in X (adata.raw is None)
        verbose=True,
    )
    res_c = adata_c.uns["liana_res"].copy()
    res_c.to_csv(os.path.join(OUTDIR, f"A0_rank_aggregate_{c}.csv"), index=False)

    # ---- A0: standard LIANA dotplot of the top interactions ------------------
    p = (
        li.pl.dotplot(
            adata=adata_c,
            colour="magnitude_rank",
            size="specificity_rank",
            inverse_colour=True,  # small ranks = strong -> bright
            inverse_size=True,  # small ranks = specific -> large
            source_labels=groups_all,
            target_labels=groups_all,
            top_n=TOP_N,
            orderby="magnitude_rank",
            orderby_ascending=True,
            figure_size=(max(10, 0.9 * len(groups_all) + 4), max(6, 0.3 * TOP_N + 2)),
        )
        + ggtitle(f"{c}: top {TOP_N} interactions (rank_aggregate)")
        + labs(colour="-log10(magnitude_rank)", size="-log10(specificity_rank)")
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

    # ---- A0: circle plot of the whole network --------------------------------
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

# Combined table (both conditions) used by the plots below
adata.uns["liana_res"] = pd.concat(res_list, ignore_index=True)
res = adata.uns["liana_res"].copy()
res.to_csv(os.path.join(OUTDIR, "A_liana_per_condition.csv"), index=False)

res["interaction"] = res["ligand_complex"] + " → " + res["receptor_complex"]
res["pair"] = res["source"].astype(str) + " → " + res["target"].astype(str)
res["sig"] = (res["magnitude_rank"] <= SIG_RANK) & (res["specificity_rank"] <= SIG_RANK)


# ---- A1: number of significant interactions per source->target ------------
def count_matrix(df, cond):
    sub = df[(df[CONDITION_KEY] == cond) & df["sig"]]
    m = pd.crosstab(sub["source"], sub["target"])
    return m


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
    im = ax.imshow(mats[c].values, cmap=cmap, vmin=0, vmax=vmax)
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
    cmap=cmap,
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

# ---- A5: circle plots per condition (overview + one page per cell type) ----
if f"{GROUP_KEY}_colors" not in adata.uns:  # consistent node colours
    sc.pl._utils._set_default_colors_for_categorical_obs(adata, GROUP_KEY)
for c in CONDITIONS:
    sig_c = res[(res[CONDITION_KEY] == c) & res["sig"]].copy()
    circle_plots(
        adata, sig_c, f"A5_circle_{c}", f"{c}: significant L-R interactions", groups
    )

# ---- A6: dotplot per source cell type -> all targets, per condition ---------
# colour = lr_means, size = -log10(cellphone_pvals); CellPhoneDB-style filter
# One multi-page PDF per condition (one page per source) + one PNG per source.
from plotnine import save_as_pdf_pages

cell_list = adata.obs[GROUP_KEY].cat.categories.tolist()
dot_dir = os.path.join(OUTDIR, "A6_dotplots_by_source")
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
                f"  A6 {c} / {celltype}: no interactions with p <= {DOTPLOT_PVAL}, skipped"
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
                target_labels=cell_list,
                filter_fun=lambda x: x["cellphone_pvals"] <= DOTPLOT_PVAL,
                top_n=DOTPLOT_TOP_N,
                orderby="lr_means" if DOTPLOT_TOP_N else None,
                orderby_ascending=False if DOTPLOT_TOP_N else None,
                figure_size=(
                    max(8, 0.45 * len(cell_list) + 4),
                    max(5, 0.25 * n_show + 2),
                ),
            )
            + ggtitle(
                f"{c}: {celltype} → all {GROUP_KEY} (cellphone_pvals ≤ {DOTPLOT_PVAL})"
            )
            + labs(size="-log10(p)", colour="lr_means")
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
        save_as_pdf_pages(
            pages,
            filename=os.path.join(OUTDIR, f"A6_dotplots_by_source_{c}.pdf"),
            verbose=False,
        )

# =============================================================================
# PART B - Pseudobulk DESeq2 (TEST vs REFERENCE) mapped to L-R pairs
# =============================================================================
print("\n[Part B] Pseudobulk differential expression ...")
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# Build pseudobulk: sum raw counts per ROI x level_2
obs = adata.obs[[SAMPLE_KEY, GROUP_KEY, CONDITION_KEY]].copy()
obs["pb_id"] = obs[SAMPLE_KEY].astype(str) + "__" + obs[GROUP_KEY].astype(str)
codes, uniques = pd.factorize(obs["pb_id"])
indicator = sparse.csr_matrix(
    (np.ones(len(codes)), (codes, np.arange(len(codes)))),
    shape=(len(uniques), len(codes)),
)
C = adata.layers["counts"]
C = C if sparse.issparse(C) else sparse.csr_matrix(C)
pb_counts = pd.DataFrame(
    np.asarray((indicator @ C).todense()).round().astype(int),
    index=uniques,
    columns=adata.var_names,
)
pb_meta = obs.drop_duplicates("pb_id").set_index("pb_id").loc[uniques]
pb_meta["n_cells"] = pd.Series(codes).value_counts().sort_index().values
pb_meta = pb_meta[pb_meta["n_cells"] >= PB_MIN_CELLS]
pb_counts = pb_counts.loc[pb_meta.index]
pb_meta.to_csv(os.path.join(OUTDIR, "B_pseudobulk_samples.csv"))

dea_list = []
for ct in groups:
    m = pb_meta[pb_meta[GROUP_KEY] == ct].copy()
    reps = m[CONDITION_KEY].value_counts()
    if (reps.reindex(CONDITIONS, fill_value=0) < PB_MIN_SAMPLES_PER_COND).any():
        print(f"  skip {ct}: replicates {reps.to_dict()}")
        continue
    X = pb_counts.loc[m.index]
    X = X.loc[:, (X >= GENE_MIN_COUNTS).sum(0) >= PB_MIN_SAMPLES_PER_COND]
    m = m.rename(columns={CONDITION_KEY: "condition"})[["condition"]]
    m["condition"] = pd.Categorical(
        m["condition"].astype(str), categories=[REFERENCE, TEST]
    )
    dds = DeseqDataSet(counts=X, metadata=m, design="~condition", quiet=True)
    dds.deseq2()
    stat = DeseqStats(dds, contrast=["condition", TEST, REFERENCE], quiet=True)
    stat.summary()
    df = stat.results_df.copy()
    df[GROUP_KEY] = ct
    dea_list.append(df)
    print(
        f"  {ct}: {len(m)} pseudobulks, {(df['padj'] < PADJ).sum()} DE genes (padj<{PADJ})"
    )

if not dea_list:
    raise RuntimeError("No cell type had enough replicates per condition for DESeq2.")

dea_df = pd.concat(dea_list)
dea_df.to_csv(os.path.join(OUTDIR, f"B_pseudobulk_DEA_{TEST}_vs_{REFERENCE}.csv"))
dea_df = dea_df.dropna(subset=["stat"])

# Map DE statistics to ligand-receptor interactions
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
)
lr_dea = lr_dea.sort_values("interaction_stat", ascending=False)
lr_dea.to_csv(
    os.path.join(OUTDIR, f"B_LR_differential_{TEST}_vs_{REFERENCE}.csv"), index=False
)

# Interactions where both partners move in the same direction and at least
# one is significant
lr_dea["both_up"] = (lr_dea["ligand_stat"] > 0) & (lr_dea["receptor_stat"] > 0)
lr_dea["both_down"] = (lr_dea["ligand_stat"] < 0) & (lr_dea["receptor_stat"] < 0)
lr_dea["any_sig"] = (lr_dea["ligand_padj"] < PADJ) | (lr_dea["receptor_padj"] < PADJ)
lr_dea["label"] = (
    lr_dea["ligand_complex"]
    + " → "
    + lr_dea["receptor_complex"]
    + "  ("
    + lr_dea["source"].astype(str)
    + " → "
    + lr_dea["target"].astype(str)
    + ")"
)
changed = lr_dea[lr_dea["any_sig"] & (lr_dea["both_up"] | lr_dea["both_down"])]
changed.to_csv(os.path.join(OUTDIR, "B_LR_changed_filtered.csv"), index=False)
print(
    f"  {len(changed)} coherently changed L-R interactions "
    f"({changed['both_up'].sum()} up in {TEST}, {changed['both_down'].sum()} down)"
)

# ---- B1: heatmap of ligand / receptor / interaction stats -----------------
# Top up (both partners up in TEST) and top down (both up in REFERENCE)
up_all = changed[changed["both_up"]].sort_values("interaction_stat", ascending=False)
down_all = changed[changed["both_down"]].sort_values("interaction_stat")
if len(up_all) + len(down_all) < 10:  # fall back to the unfiltered ranking
    up_all = lr_dea.sort_values("interaction_stat", ascending=False)
    down_all = lr_dea.sort_values("interaction_stat")
top_b = (
    pd.concat([up_all.head(TOP_N // 2), down_all.head(TOP_N // 2)])
    .drop_duplicates("label")
    .sort_values("interaction_stat")
)

vals = top_b[["ligand_stat", "receptor_stat", "interaction_stat"]].values
pvals = top_b[["ligand_padj", "receptor_padj"]].values
lim = np.nanmax(np.abs(vals)) or 1
fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(top_b) + 1.8))
im = ax.imshow(vals, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
for i in range(len(top_b)):
    for j in range(2):
        if pvals[i, j] < PADJ:
            ax.text(j, i, "*", ha="center", va="center", fontsize=9)
ax.set_xticks(range(3))
ax.set_xticklabels(["Ligand\n(source)", "Receptor\n(target)", "Interaction"])
ax.set_yticks(range(len(top_b)))
ax.set_yticklabels(top_b["label"], fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.4, label=f"Wald stat ({TEST} vs {REFERENCE})")
ax.set_title(f"Top differential L-R interactions\n* padj < {PADJ}")
savefig(fig, "B1_heatmap_top_changed")

# ---- B2: diverging bar of interaction_stat --------------------------------
fig, ax = plt.subplots(figsize=(8, 0.28 * len(top_b) + 1.5))
tb = top_b.sort_values("interaction_stat")
ax.barh(
    tb["label"],
    tb["interaction_stat"],
    color=np.where(tb["interaction_stat"] > 0, "#c0392b", "#2c6fbb"),
)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel(
    f"Interaction stat (mean of ligand & receptor Wald stat, {TEST} vs {REFERENCE})"
)
ax.set_title(
    f"Differential L-R interactions\n(red = up in {TEST}, blue = up in {REFERENCE})"
)
ax.tick_params(axis="y", labelsize=8)
savefig(fig, "B2_interaction_stat_barplot")

# ---- B3: ligand vs receptor stat scatter ----------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(
    lr_dea["ligand_stat"], lr_dea["receptor_stat"], s=5, c="lightgrey", linewidths=0
)
up = changed[changed["both_up"]]
down = changed[changed["both_down"]]
ax.scatter(
    up["ligand_stat"], up["receptor_stat"], s=10, c="#c0392b", label=f"up in {TEST}"
)
ax.scatter(
    down["ligand_stat"],
    down["receptor_stat"],
    s=10,
    c="#2c6fbb",
    label=f"up in {REFERENCE}",
)
ax.axhline(0, c="k", lw=0.6)
ax.axvline(0, c="k", lw=0.6)
for _, r in pd.concat([up.head(6), down.tail(6)]).iterrows():
    ax.annotate(
        r["ligand_complex"] + "→" + r["receptor_complex"],
        (r["ligand_stat"], r["receptor_stat"]),
        fontsize=6,
    )
ax.set_xlabel("Ligand stat (in source)")
ax.set_ylabel("Receptor stat (in target)")
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"{TEST} vs {REFERENCE}")
savefig(fig, "B3_ligand_vs_receptor_stat")

# ---- B4: source -> target summary of changed interactions ------------------
fig, axes = plt.subplots(
    1, 2, figsize=(2 * max(5, 0.45 * n + 2), max(4.5, 0.45 * n + 1.5))
)
for ax, (lab, sub, cmap) in zip(
    axes, [(f"Up in {TEST}", up, "Reds"), (f"Up in {REFERENCE}", down, "Blues")]
):
    mat = pd.crosstab(sub["source"], sub["target"]).reindex(
        index=groups, columns=groups, fill_value=0
    )
    im = ax.imshow(mat.values, cmap=cmap)
    ax.set_title(f"# L-R interactions {lab}")
    ax.set_xticks(range(n))
    ax.set_xticklabels(groups, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(groups)
    ax.set_xlabel("Target (receptor)")
    ax.set_ylabel("Source (ligand)")
    fig.colorbar(im, ax=ax, shrink=0.7)
savefig(fig, "B4_changed_interactions_source_target")

# ---- B5: circle plots of differential interactions -------------------------
circle_plots(
    adata,
    up.copy(),
    f"B5_circle_up_in_{TEST}",
    f"L-R interactions up in {TEST} vs {REFERENCE}",
    groups,
)
circle_plots(
    adata,
    down.copy(),
    f"B5_circle_up_in_{REFERENCE}",
    f"L-R interactions up in {REFERENCE} vs {TEST}",
    groups,
)


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
import pycrosstalker as ct
import pycrosstalker.plots as ctpl

print("\n[Part C] pyCrossTalkeR ...")
CMP = f"{TEST}_x_{REFERENCE}"  # e.g. "IPF_x_PM08"

lr_in = res.copy()
lr_in["ligand"] = lr_in["ligand_complex"]  # from_liana expects a 'ligand' column
lr_in[CONDITION_KEY] = lr_in[CONDITION_KEY].astype(str)
ctk = ad.AnnData(uns={"liana_res": lr_in})
ctk = ct.tl.from_liana(
    ctk,
    liana_key="liana_res",
    score_key="lr_means",
    pval_key="cellphone_pvals",
    pval_filter=True,
    condition_key=CONDITION_KEY,
)

ct_colors = dict(
    zip(adata.obs[GROUP_KEY].cat.categories, adata.uns[f"{GROUP_KEY}_colors"])
)  # same colours as other plots
ctk_dir = os.path.join(OUTDIR, "C_crosstalker")
os.makedirs(ctk_dir, exist_ok=True)
ctk = ct.tl.analise_LR(
    ctk,
    out_path=ctk_dir,
    colors=ct_colors,
    comparison=[(TEST, REFERENCE)],  # (experiment, control)
    filename="crosstalker",
    save=False,
)
ctk_res = ctk.uns["pycrosstalker"]["results"]

# Save tables
for key, rk in ctk_res["rankings"].items():
    rk.to_csv(os.path.join(ctk_dir, f"C_ranking_{key}.csv"), index=False)
ctk_res["stats"][CMP].to_csv(os.path.join(ctk_dir, f"C_fisher_{CMP}.csv"), index=False)
ctk_res["stats"][f"{CMP}:MannU"].to_csv(
    os.path.join(ctk_dir, f"C_mannwhitney_{CMP}.csv"), index=False
)
ctk_res["graphs"][CMP].to_csv(
    os.path.join(ctk_dir, f"C_graph_edges_{CMP}.csv"), index=False
)


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


# ---- C1: CCI graphs (each condition + comparison) --------------------------
for c in CONDITIONS:
    crosstalker_cci(c, f"{c}")
crosstalker_cci(
    CMP,
    f"Comparative ({TEST} vs {REFERENCE})\n"
    f"red = higher in {TEST}, blue = higher in {REFERENCE}",
    signed=True,
)

# ---- C2: differential PageRank per cell type -------------------------------
# PageRank in the comparison ranking = PageRank(TEST) - PageRank(REFERENCE).
# (Influencer/Listener are computed on the difference graph itself, not as a
#  difference between conditions, so they are kept in the CSV but not plotted.)
rk = ctk_res["rankings"][CMP].set_index("nodes").sort_values("Pagerank")
v = rk["Pagerank"]
fig, ax = plt.subplots(figsize=(6, 0.35 * len(rk) + 1.8))
ax.barh(v.index, v, color=np.where(v > 0, "#c0392b", "#2c6fbb"))
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel(f"Δ PageRank ({TEST} − {REFERENCE})")
ax.set_ylabel(GROUP_KEY)
ax.set_title(
    f"CrossTalkeR: change in cell-type importance\n"
    f"red = more central in {TEST}, blue = more central in {REFERENCE}"
)
savefig(fig, os.path.join("C_crosstalker", f"C2_pagerank_{CMP}"))


# ---- C3: Fisher (number) and Mann-Whitney (strength) per cell pair ---------
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


pair_heatmap(
    ctk_res["stats"][CMP],
    "lodds",
    "p_value",
    f"Number of interactions: log odds {TEST} vs {REFERENCE} (Fisher)",
    f"C3_fisher_{CMP}",
)
pair_heatmap(
    ctk_res["stats"][f"{CMP}:MannU"],
    "lfc",
    "p_value",
    f"Interaction strength: log fold change {TEST} vs {REFERENCE} (Mann-Whitney)",
    f"C3_mannwhitney_{CMP}",
)

# =============================================================================
# PART D - Spatial maps of the top differential L-R pairs
# =============================================================================
# For each top pair from Part B (both partners changed in the same direction),
# LIANA's bivariate metric scores, for every cell, how strongly the ligand and
# the receptor are co-expressed within its spatial neighbourhood (Gaussian
# kernel, SPATIAL_BANDWIDTH um). Neighbourhoods are built per ROI, because each
# ROI has its own coordinate frame. The local score (cosine similarity) is
# mapped onto the tissue, and summarised per ROI so the two conditions can be
# compared with ROIs as replicates.
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu

print("\n[Part D] Spatial L-R maps ...")
sp_dir = OUTDIR / "D_spatial"
sp_dir.mkdir(exist_ok=True)
COND_COLORS = {TEST: "#c0392b", REFERENCE: "#2c6fbb"}
ARROW = " \u2192 "


def pretty(pair):
    return pair.replace("^", ARROW)


# ---- choose pairs: top up and top down (unique ligand-receptor) ------------
def top_pairs(df, ascending, n):
    d = df.sort_values("interaction_stat", ascending=ascending)
    d = d.drop_duplicates(["ligand_complex", "receptor_complex"])
    return d.head(n)


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
    [
        "pair",
        "direction",
        "source",
        "target",
        "interaction_stat",
        "ligand_padj",
        "receptor_padj",
    ]
].to_csv(sp_dir / "D_selected_pairs.csv", index=False)
print("  Pairs mapped:", ", ".join(sel["pair"]))

resource_sel = sel[["ligand_complex", "receptor_complex"]].rename(
    columns={"ligand_complex": "ligand", "receptor_complex": "receptor"}
)
genes_needed = sorted(
    {
        g
        for x in pd.concat([resource_sel["ligand"], resource_sel["receptor"]])
        for g in x.split("_")
    }
    & set(adata.var_names)
)

# ---- run bivariate per ROI --------------------------------------------------
roi_meta = (
    adata.obs[[SAMPLE_KEY, CONDITION_KEY]]
    .drop_duplicates()
    .set_index(SAMPLE_KEY)[CONDITION_KEY]
    .astype(str)
)
roi_order = (
    roi_meta.sort_values(key=lambda c: c.map({TEST: 0, REFERENCE: 1}))
    .index.astype(str)
    .tolist()
)  # TEST ROIs first

local_scores, coords, celltypes, global_rows = {}, {}, {}, []
for roi in roi_order:
    a = adata[adata.obs[SAMPLE_KEY] == roi, genes_needed].copy()
    if a.n_obs < 50:
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
            resource=resource_sel,
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
    celltypes[roi] = a.obs[GROUP_KEY].astype(str).values
    for pair, row in lrd.var.iterrows():
        global_rows.append(
            {
                SAMPLE_KEY: roi,
                CONDITION_KEY: roi_meta[roi],
                "pair": pair,
                "morans_I": row["morans"],
                "mean_local_score": local_scores[roi][pair].mean(),
            }
        )
    print(f"  {roi}: {a.n_obs} cells, {lrd.n_vars} pairs scored")

roi_summary = pd.DataFrame(global_rows)
roi_summary.to_csv(sp_dir / "D_per_ROI_summary.csv", index=False)

# Mean local score per level_2 cell type and condition (where does it happen?)
ct_rows = []
for roi, df in local_scores.items():
    d = df.copy()
    d[GROUP_KEY] = celltypes[roi]
    d[CONDITION_KEY] = roi_meta[roi]
    ct_rows.append(d)
if ct_rows:
    (
        pd.concat(ct_rows)
        .groupby([CONDITION_KEY, GROUP_KEY])
        .mean()
        .to_csv(sp_dir / "D_mean_local_score_by_celltype.csv")
    )

pairs_scored = [
    p for p in sel["pair"] if p in roi_summary.get("pair", pd.Series()).values
]

# ---- D1: tissue maps, one page per pair, all ROIs ---------------------------
rois_done = [r for r in roi_order if r in local_scores]
ncol = min(4, max(1, len(rois_done)))
nrow = int(np.ceil(len(rois_done) / ncol)) if rois_done else 0
with PdfPages(sp_dir / "D1_spatial_maps_all_pairs.pdf") as pdf:
    for pair in pairs_scored:
        vals_all = np.concatenate(
            [local_scores[r][pair].values for r in rois_done if pair in local_scores[r]]
        )
        vmax = np.nanpercentile(vals_all, 99) or 1
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False
        )
        for ax in axes.flat:
            ax.axis("off")
        for ax, roi in zip(axes.flat, rois_done):
            xy = coords[roi]
            if pair in local_scores[roi]:
                v = local_scores[roi][pair].values
                o = np.argsort(v)  # draw hotspots on top
                sca = ax.scatter(
                    xy[o, 0],
                    xy[o, 1],
                    c=v[o],
                    s=SPATIAL_POINT_SIZE,
                    cmap=cmap,  # not sure is it should be divergent colors
                    vmin=0,
                    vmax=vmax,
                    linewidths=0,
                    rasterized=True,
                )
            else:
                ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    c="lightgrey",
                    s=SPATIAL_POINT_SIZE,
                    linewidths=0,
                    rasterized=True,
                )
                ax.text(
                    0.5,
                    0.02,
                    "not expressed",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=7,
                )
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_title(
                f"{roi} ({roi_meta[roi]})",
                fontsize=9,
                color=COND_COLORS.get(roi_meta[roi], "k"),
            )
        info = sel.set_index("pair").loc[pair]
        fig.suptitle(
            f"{pretty(pair)}  ({info['direction']}; "
            f"{info['source']} → {info['target']})\n"
            f"local L-R co-expression (cosine, bandwidth {SPATIAL_BANDWIDTH} µm)",
            fontsize=11,
        )
        fig.colorbar(sca, ax=axes.ravel().tolist(), shrink=0.5, label="local score")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", pair)
        fig.savefig(sp_dir / f"D1_spatial_{safe}.png", dpi=300)
        pdf.savefig(fig)
        plt.close(fig)

# ---- D2: mean local score per ROI, IPF vs PM08 ------------------------------
if pairs_scored:
    ncol2 = min(5, len(pairs_scored))
    nrow2 = int(np.ceil(len(pairs_scored) / ncol2))
    fig, axes = plt.subplots(
        nrow2, ncol2, figsize=(2.6 * ncol2, 3.2 * nrow2), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    rng = np.random.default_rng(0)
    for ax, pair in zip(axes.flat, pairs_scored):
        ax.axis("on")
        d = roi_summary[roi_summary["pair"] == pair]
        groups_v = []
        for i, cnd in enumerate([REFERENCE, TEST]):
            y = d.loc[d[CONDITION_KEY] == cnd, "mean_local_score"].values
            groups_v.append(y)
            ax.scatter(
                i + rng.uniform(-0.12, 0.12, len(y)),
                y,
                s=25,
                color=COND_COLORS[cnd],
                edgecolors="k",
                linewidths=0.3,
            )
            if len(y):
                ax.hlines(np.median(y), i - 0.25, i + 0.25, color="k", lw=1.2)
        p = (
            mannwhitneyu(*groups_v).pvalue
            if all(len(g) >= 2 for g in groups_v)
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
    savefig(fig, "D_spatial/D2_per_ROI_local_score")

    # ---- D3: global Moran's I (spatial clustering) per pair x ROI ----------
    m = (
        roi_summary.pivot(index="pair", columns=SAMPLE_KEY, values="morans_I")
        .reindex(index=pairs_scored, columns=rois_done)
        .astype(float)
    )
    fig, ax = plt.subplots(
        figsize=(0.5 * len(rois_done) + 4, 0.4 * len(pairs_scored) + 2)
    )
    lim = np.nanmax(np.abs(m.values)) or 1
    im = ax.imshow(m.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(rois_done)))
    ax.set_xticklabels(rois_done, rotation=90, fontsize=8)
    for t, roi in zip(ax.get_xticklabels(), rois_done):
        t.set_color(COND_COLORS.get(roi_meta[roi], "k"))
    ax.set_yticks(range(len(pairs_scored)))
    ax.set_yticklabels([p.replace("^", " → ") for p in pairs_scored], fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Moran's I (bivariate)")
    ax.set_title(
        f"Global spatial co-localisation of ligand & receptor\n"
        f"(ROI labels: red = {TEST}, blue = {REFERENCE})"
    )
    savefig(fig, "D_spatial/D3_morans_I_heatmap")

print(f"\nDone. Results in: {os.path.abspath(OUTDIR)}")
