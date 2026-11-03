"""Create plotes for CCC analysis of COPD vs MICA dataset."""

import os
from pathlib import Path

import anndata as ad
import liana as li
import matplotlib.pyplot as plt
import pandas as pd

INPUT_DIR = Path("/Volumes/sep22/ephemeral/output/ccc/")

# Output directories for plots
OUTPUT_DIR = Path("/Volumes/sep22/ephemeral/output/ccc/plots/MANUAL_MICA_COPD")
os.makedirs(OUTPUT_DIR, exist_ok=True)

adata_COPD = ad.read_zarr(INPUT_DIR / "COPDvMICA/ccc_COPD.zarr")

adata_MICA = ad.read_zarr(INPUT_DIR / "COPDvMICA/ccc_MICA.zarr")


# Res for COPD and MICA separately, with condition column added
res_COPD = adata_COPD.uns["liana_res"].assign(condition="COPD")
res_MICA = adata_MICA.uns["liana_res"].assign(condition="MICA")

# Extract labels from your condition-specific results
source_labels_COPD = list(set(res_COPD["source"].dropna().unique()))
target_labels_COPD = list(set(res_COPD["target"].dropna().unique()))

source_labels_MICA = list(set(res_MICA["source"].dropna().unique()))
target_labels_MICA = list(set(res_MICA["target"].dropna().unique()))

# Combined results from both conditions
res_all = pd.concat(
    [
        adata_COPD.uns["liana_res"].assign(condition="COPD"),
        adata_MICA.uns["liana_res"].assign(condition="MICA"),
    ],
    ignore_index=True,
)


wide = res_all.pivot_table(
    index=["source", "target", "ligand_complex", "receptor_complex"],
    columns="condition",
    values="magnitude_rank",
)

# Extract labels from your condition-specific results
source_labels = list(set(res_all["source"].dropna().unique()))
target_labels = list(set(res_all["target"].dropna().unique()))

# Rename target to include condition for better visualization in dotplot
res_all["target"] = res_all["target"] + " - " + res_all["condition"] + ""
target_labels = list(res_all["target"].unique())

# replace MICA with Proximal control in the target labels for clarity
res_all["target"] = res_all["target"].str.replace(" - MICA", " - Control", regex=False)
res_all["target"] = res_all["target"].str.replace(
    " - COPD", " - COPD", regex=False
)  # no-op, for clarity

res_all = pd.concat(
    [
        adata_MICA.uns["liana_res"].assign(condition=" Proximal control"),
        adata_COPD.uns["liana_res"].assign(condition="COPD"),
    ],
    ignore_index=True,
)
res_all["target"] = res_all["target"] + " " + res_all["condition"] + ""


# Create dotplots for each source cell type for both conditions
for celltype in source_labels:
    print(f"Generating dotplot for source cell type: {celltype}")
    p = li.pl.dotplot(
        liana_res=res_all,
        colour="magnitude_rank",
        size="specificity_rank",
        inverse_colour=True,  # small ranks = strong -> bright
        inverse_size=True,  # small ranks = specific -> large
        # only cell types that appear in this condition's results
        source_labels=celltype,
        top_n=25,
        orderby="magnitude_rank",
        orderby_ascending=True,
        cmap="Blues",
        figure_size=(12, 10),
    )
    p.save(
        OUTPUT_DIR / f"dotplot_{celltype.replace(' ', '_').replace('/', '_')}.pdf",
        dpi=300,
    )


# Create plots for each condition separately
dict_condition = {
    "COPD": (adata_COPD, res_COPD, source_labels_COPD, source_labels_COPD),
    "MICA": (adata_MICA, res_MICA, source_labels_MICA, source_labels_MICA),
}

for condition, (adata_cond, res_cond, source_labels_cond) in dict_condition.items():
    print(f"Processing condition: {condition}")
    OUTPUT_DIR_CONDITION = OUTPUT_DIR / condition
    os.makedirs(OUTPUT_DIR_CONDITION, exist_ok=True)
    for celltype in source_labels_cond:
        celltype_safe = celltype.replace(" ", "_").replace("/", "_")

        print(f"Plotting source cell type: {celltype} in {condition} condition")
        # Dotplot
        OUTPUT_DOT = OUTPUT_DIR_CONDITION / "dotplots"
        os.makedirs(OUTPUT_DOT, exist_ok=True)
        p = li.pl.dotplot(
            adata=adata_cond,
            liana_res=res_cond,
            colour="magnitude_rank",
            size="specificity_rank",
            inverse_colour=True,  # small ranks = strong -> bright
            inverse_size=True,  # small ranks = specific -> large
            source_labels=[celltype],
            top_n=25,
            orderby="magnitude_rank",
            orderby_ascending=True,
            cmap="Blues",
            figure_size=(10, 10),
        )
        p.save(OUTPUT_DOT / f"dotplot_{condition}_{celltype_safe}.pdf")

        OUTPUT_DOT_FILTERED = OUTPUT_DIR_CONDITION / "dotplots_filtered"
        os.makedirs(OUTPUT_DOT_FILTERED, exist_ok=True)
        p = li.pl.dotplot(
            adata=adata_cond,
            liana_res=res_cond,
            colour="magnitude_rank",
            size="specificity_rank",
            inverse_colour=True,  # small ranks = strong -> bright
            inverse_size=True,  # small ranks = specific -> large
            source_labels=[celltype],
            top_n=25,
            orderby="magnitude_rank",
            orderby_ascending=True,
            cmap="Blues",
            figure_size=(10, 10),
            filter_fn=lambda x: x["specificity_rank"] <= 0.05,
        )
        p.save(
            OUTPUT_DOT_FILTERED / f"dotplot_{condition}_{celltype_safe}_filtered.pdf"
        )

        # Circle plot
        OUTPUT_CIRCLE = OUTPUT_DIR_CONDITION / "circleplots"
        os.makedirs(OUTPUT_CIRCLE, exist_ok=True)
        ax = li.pl.circle_plot(
            adata=adata_cond,
            pivot_mode="counts",  # edge weights are counts of interactions
            groupby="level_2",
            source_labels=[celltype],  # None = whole network
            figure_size=(12, 12),
            edge_alpha=0.8,
            edge_arrow_size=10,
            node_size_scale=(50, 400),
            edge_width_scale=(0.5, 8),
            node_label_offset=(0.1, -0.2),
            node_label_size=6,
        )
        ax.set_title(
            f"Circle plot for source cell type: {celltype} in {condition} condition",
            fontsize=14,
        )
        ax.get_figure().savefig(
            OUTPUT_CIRCLE / f"circleplot_{condition}_{celltype_safe}.pdf"
        )
        plt.close(ax.get_figure())

        # Filtered circle plot for top interactions based on specificity rank
        OUTPUT_CIRCLE_FILTERED = OUTPUT_DIR_CONDITION / "circleplots_filtered"
        os.makedirs(OUTPUT_CIRCLE_FILTERED, exist_ok=True)
        try:
            ax = li.pl.circle_plot(
                adata=adata_cond,
                groupby="level_2",
                source_labels=[celltype],
                pivot_mode="counts",
                filter_fn=lambda x: x["specificity_rank"] <= 0.05,
                figure_size=(10, 10),
                edge_alpha=0.8,
                edge_arrow_size=10,
                node_size_scale=(50, 400),
                edge_width_scale=(0.5, 8),
                node_label_offset=(0.1, -0.2),
                node_label_size=8,
            )
            ax.set_title(
                f"Filtered circle plot: {celltype} in {condition} condition",
                fontsize=14,
            )
            ax.get_figure().savefig(
                OUTPUT_CIRCLE_FILTERED / f"circleplot_{condition}_{celltype_safe}.pdf"
            )
            plt.show()
            plt.close(ax.get_figure())
        except ValueError as e:
            print(f"Skipping {celltype}: {e}")
            plt.close("all")
            continue
