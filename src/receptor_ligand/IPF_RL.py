"""Cell-cell communication (ligand-receptor) analysis: IPF vs PM08.

Two complementary analyses, both grouped by adata.obs["level_2"]:

  Part A - Per-condition LIANA consensus (rank_aggregate run separately for
           IPF and PM08). Answers: which interactions are strong/specific in
           each disease, and how does their magnitude (lr_means) shift?
           Descriptive; does not account for sample-to-sample variation.

  Part B - Sample-aware differential analysis. Pseudobulk (ROI x level_2),
           PyDESeq2 PM08 vs IPF per cell type, then LIANA maps the DE
           statistics onto ligand-receptor pairs (li.multi.df_to_lr).
           This is the approach to use for statistical claims.

Requirements (tested with liana 1.10, pydeseq2 0.5.4, scanpy 1.11):
    pip install liana pydeseq2 scanpy

Outputs go to OUTDIR (CSV tables + PDF/PNG figures).
"""

import os
import warnings
from pathlib import Path

import anndata as ad
import liana as li
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

warnings.filterwarnings("ignore", category=FutureWarning)

# CONFIG - edit these
INPUT_H5AD = "path/to/your_data.h5ad"
DIR = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output"
)
OUTDIR = DIR / "ccc_IPF_vs_PM08"

CONDITION_KEY = "condition"
CONDITIONS = ["IPF", "PM08"]
REFERENCE = "IPF"  # log fold changes are PM08 relative to IPF
TEST = "PM08"

GROUP_KEY = "level_2"  # cell types used as sources/targets
SAMPLE_KEY = "ROI"  # replicate unit for pseudobulk (Part B)
DROP_ROIS = ["PM08_159"]

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

os.makedirs(OUTDIR, exist_ok=True)
sc.settings.figdir = OUTDIR
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


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
        fig.savefig(os.path.join(OUTDIR, f"{name}_overview.png"), dpi=200)
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
adata = ad.read_zarr(DIR / "AIRSCAPE/adata_final_object/adata_with_metadata.zarr")

# Subset to IPF and PM08 samples
adata = adata[adata.obs[CONDITION_KEY].isin(CONDITIONS)]
# Exclude PM08_159
adata = adata[
    ~adata.obs[SAMPLE_KEY].isin(DROP_ROIS)
].copy()  # .copy() avoids view warnings

# Remove alveolar fibroblasts (collagen high) because they are only present in one donor
cell_types_to_remove = [
    "Alveolar fibroblasts (collagen high)",
]
adata = adata[~adata.obs[GROUP_KEY].isin(cell_types_to_remove)].copy()

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
print("\n[Part A] Running LIANA rank_aggregate per condition ...")
li.mt.rank_aggregate.by_sample(
    adata,
    sample_key=CONDITION_KEY,  # run once per condition
    groupby=GROUP_KEY,
    resource_name=RESOURCE,
    expr_prop=EXPR_PROP,
    min_cells=MIN_CELLS_PER_GROUP,
    n_perms=N_PERMS,
    use_raw=False,
    key_added="liana_res",
    verbose=True,
)
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
    im = ax.imshow(mats[c].values, cmap="Blues", vmin=0, vmax=vmax)
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

# ---- A3: IPF vs PM08 scatter of lr_means --------------------------------------
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
    cmap="viridis",
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

# =============================================================================
# PART B - Pseudobulk DESeq2 (PM08 vs IPF) mapped to L-R pairs
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
dea_df.to_csv(os.path.join(OUTDIR, "B_pseudobulk_DEA_PM08_vs_IPF.csv"))
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
lr_dea.to_csv(os.path.join(OUTDIR, "B_LR_differential_PM08_vs_IPF.csv"), index=False)

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

print(f"\nDone. Results in: {os.path.abspath(OUTDIR)}")
