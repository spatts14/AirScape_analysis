"""Plot 16S mean expression and percent positive per cell type and condition as a dot plot."""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import stats

base_dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output"
)
save_fig = base_dir / "16S/figures"


adata = sc.read_h5ad(
    base_dir / "2026-03-27_analysis_run/subset_adata/adata_subset_IPF_PM08.h5ad"
)


adata = adata[~adata.obs["level_1"].str.contains("Unknown", case=False)].copy()
gene = "16S"
level = "level_1"
condition_order = ["PM08", "IPF"]
cmap = sns.diverging_palette(220, 20, as_cmap=True)


# --- 1. Compute per (cell_type, condition) stats ---
df = adata.obs[[level, "condition"]].copy()
df["condition"] = pd.Categorical(
    df["condition"], categories=condition_order, ordered=True
)
df["expr"] = adata[:, gene].X.toarray().flatten()

stats = (
    df.groupby([level, "condition"])
    .agg(mean_expr=("expr", "mean"), pct_pos=("expr", lambda x: (x > 0).mean() * 100))
    .reset_index()
)

# --- 2. Pivot for plotting ---
mean_pivot = stats.pivot(index=level, columns="condition", values="mean_expr")
pct_pivot = stats.pivot(index=level, columns="condition", values="pct_pos")

cell_types = mean_pivot.index.tolist()
conditions = mean_pivot.columns.tolist()

n_conditions = len(conditions)
n_celltypes = len(cell_types)

# --- 3. Build dot plot ---
spacing = 0.3  # x inches per condition
y_spacing = 0.4  # y inches per cell type
x_margin = 2.5  # extra width for colorbar + legend
y_margin = 1.2  # extra height for labels + padding

x_positions = [j * spacing for j in range(n_conditions)]
y_positions = [i * y_spacing for i in range(n_celltypes)]

fig_width = n_conditions * spacing + x_margin
fig_height = n_celltypes * y_spacing + y_margin

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# vmin = stats["mean_expr"].min()
# vmax = stats["mean_expr"].max()
vmin = -0.2
vmax = 0.2
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

max_size = 600

for i, ct in enumerate(cell_types):
    for j, cond in enumerate(conditions):
        mean_val = mean_pivot.loc[ct, cond]
        pct_val = pct_pivot.loc[ct, cond]

        if pd.isna(mean_val):
            continue

        size = (pct_val / 100) * max_size
        color = cmap(norm(mean_val))

        ax.scatter(
            x_positions[j],
            y_positions[i],
            s=size,
            color=color,
            edgecolors="grey",
            linewidths=0.4,
        )

# --- 4. Axes formatting ---
ax.set_xticks(x_positions)
ax.set_xticklabels(conditions, rotation=0, ha="center", fontsize=12)
ax.set_yticks(y_positions)
ax.set_yticklabels(cell_types, fontsize=12)

ax.set_xlim(x_positions[0] - 0.3, x_positions[-1] + 0.3)
ax.set_ylim(y_positions[0] - 0.3, y_positions[-1] + 0.3)

ax.set_facecolor("white")
sns.despine(ax=ax, left=True, bottom=True)

# --- 5. Colorbar ---
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.05, aspect=12)
cbar.set_label("Mean expression", fontsize=10)
cbar.ax.tick_params(labelsize=12)

# --- 6. Size legend ---
legend_pcts = [25, 50, 75, 100]
for pct in legend_pcts:
    ax.scatter(
        [],
        [],
        s=(pct / 100) * max_size,
        color="grey",
        edgecolors="grey",
        linewidths=0.4,
        label=f"{pct}%",
    )
ax.legend(
    title="percent 16S+ cells",
    title_fontsize=10,
    fontsize=10,
    bbox_to_anchor=(1.3, 1),
    loc="upper left",
    frameon=False,
)

plt.tight_layout()
plt.savefig(save_fig / f"_16S_dotplot_{level}.pdf", dpi=150, bbox_inches="tight")
plt.show()
