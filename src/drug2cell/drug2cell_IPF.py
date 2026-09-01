"""Drug2Cell for IPF data."""

# Load packages
import gc
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import squidpy as sq

import drug2cell as d2c


# Define function
def plot_top_drugs_by_condition_for_all_celltypes(
    d2c_adata,
    level_col,
    rank_key,
    output_dir,
    groupby="condition",
    group_order=None,
    n_top=10,
    cmap=sns.color_palette("Blues", as_cmap=True),
):
    """Loop over every cell type in level and plot top ranked drugs split by condition.

    Loop over every cell type in a given level, plotting its top ranked
    drugs (pulled from an existing rank_genes_groups result) split by condition.

    Args:
        d2c_adata : AnnData
            The drug2cell-scored AnnData (e.g. adata.uns['drug2cell']).
        level_col : str
            The obs column defining cell types to loop over, e.g. "level_1",
            "level_2", or "level_3".
        rank_key : str
            The .uns key under which rank_genes_groups results (grouped by
            level_col) were stored, e.g. "d2c_rank_genes_groups_level_3".
        output_dir : Path
            Base output directory; a level_col subfolder is created inside it.
        groupby : str
            Column to split the dotplot by within each cell type, default "condition".
        group_order : list, optional
            Explicit category order for groupby (e.g. ["PM08", "IPF"]). If None,
            uses whatever order is already set on the column.
        n_top : int
            Number of top drugs to show per cell type.
        cmap : colormap
            Colormap for the dotplot.

    Returns:
        dict mapping cell type -> list of top drug names plotted.
    """
    n_missing = d2c_adata.obs[level_col].isna().sum()
    if n_missing > 0:
        print(
            f"Warning: {n_missing} cells have missing '{level_col}' and are excluded."
        )

    celltypes = d2c_adata.obs[level_col].dropna().unique().tolist()

    results = {}

    # Create a per-level subdirectory and point scanpy's figdir at it for
    output_dir_level = output_dir / level_col
    output_dir_level.mkdir(exist_ok=True, parents=True)
    original_figdir = sc.settings.figdir
    sc.settings.figdir = output_dir_level

    try:
        for celltype in celltypes:
            print(f"Plotting top {n_top} drugs for '{celltype}'...")

            top_drugs = (
                sc.get.rank_genes_groups_df(d2c_adata, group=celltype, key=rank_key)
                .head(n_top)["names"]
                .tolist()
            )

            if not top_drugs:
                print(f"Skipping '{celltype}': no ranked drugs found.")
                continue

            mask = d2c_adata.obs[level_col] == celltype
            subset = d2c_adata[mask].copy()

            # Apply the requested category order to the groupby column
            if group_order is not None:
                subset.obs[groupby] = subset.obs[groupby].astype("category")
                present = [
                    g for g in group_order if g in subset.obs[groupby].cat.categories
                ]
                remaining = [
                    g for g in subset.obs[groupby].cat.categories if g not in present
                ]
                subset.obs[groupby] = subset.obs[groupby].cat.reorder_categories(
                    present + remaining
                )

            safe_name = str(celltype).replace(" ", "_").replace("/", "-")

            sc.pl.dotplot(
                subset,
                var_names=top_drugs,
                groupby=groupby,
                cmap=cmap,
                save=f"{safe_name}_d2c_top_drugs_by_{groupby}.pdf",
            )

            results[celltype] = top_drugs
    finally:
        sc.settings.figdir = original_figdir

    return results


# Set working directory
dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output"
)

# Set output dir
output = dir / "drug2cell" / "output"
output.mkdir(exist_ok=True, parents=True)
sc.settings.figdir = output  # set figure directory

# Set colors
cmap = sns.color_palette("Blues", as_cmap=True)

# Variables
drugs_of_interest = [
    "CHEMBL1256391|PIRFENIDONE",
]

# Load adata
adata = ad.read_zarr(dir / "AIRSCAPE/adata_final_object/adata_with_metadata.zarr")

# Subset to IPF and PM08 samples
adata = adata[adata.obs["condition"].isin(["IPF", "PM08"])]

# Calculate drug2cell scores
d2c.score(adata, use_raw=True)

# Save drug2cell-scored object to adata.uns
d2c_adata = adata.uns["drug2cell"]
del adata
gc.collect()

# Export drug2cell scores to Excel
drugs_present = d2c_adata.var
drugs_present.reset_index().to_excel(output / "drug2cell_scores.xlsx", index=False)


# Calculate differentially expressed drug2cell scores for each cell type
sc.tl.rank_genes_groups(
    d2c_adata,
    method="wilcoxon",
    groupby="level_2",
    key_added="d2c_rank_genes_groups_level_2",
)

sc.tl.rank_genes_groups(
    d2c_adata,
    method="wilcoxon",
    groupby="level_3",
    key_added="d2c_rank_genes_groups_level_3",
)

# Plot differentially expressed drug2cell scores for each cell type
sc.pl.rank_genes_groups_dotplot(
    d2c_adata,
    key="d2c_rank_genes_groups_level_2",
    swap_axes=True,
    dendrogram=False,
    n_genes=5,
    cmap=cmap,
    save="drug2cell_rank_genes_groups_dotplot_level_2.pdf",
)

sc.pl.rank_genes_groups_dotplot(
    d2c_adata,
    key="d2c_rank_genes_groups_level_3",
    swap_axes=True,
    dendrogram=False,
    n_genes=5,
    cmap=cmap,
    save="drug2cell_rank_genes_groups_dotplot_level_3.pdf",
)

# Dotplot for each cell type
top_drugs_by_celltype_l2 = plot_top_drugs_by_condition_for_all_celltypes(
    d2c_adata,
    level_col="level_2",
    rank_key="d2c_rank_genes_groups_level_2",
    output_dir=output,
    group_order=["PM08", "IPF"],
)

top_drugs_by_celltype_l3 = plot_top_drugs_by_condition_for_all_celltypes(
    d2c_adata,
    level_col="level_3",
    rank_key="d2c_rank_genes_groups_level_3",
    output_dir=output,
    group_order=["PM08", "IPF"],
)

# Plot list of specified drugs of interest across data
# UMAP
sc.pl.umap(
    d2c_adata,
    color=[*drugs_of_interest, "level_2"],
    color_map=cmap,
    save="drug2cell_umap_drugs_of_interest.pdf",
)

# Spatial plots
# Subset from the drug2cell-scored object (d2c_adata), not the original adata,
# since drug identifiers only exist as var_names on d2c_adata, and its .obs/.obsm
# carry ROI, condition, and spatial coordinates through from the original adata.
assert "ROI" in d2c_adata.obs.columns, "ROI column missing from drug2cell adata.obs"
assert "spatial" in d2c_adata.obsm, (
    "spatial coordinates missing from drug2cell adata.obsm"
)

rois = d2c_adata.obs["ROI"].unique().tolist()

for roi in rois:
    print(f"Plotting ROI: {roi}")
    # Using a view (no .copy()) since we only read from roi_adata here;
    # switch to .copy() if squidpy throws a SettingWithCopyWarning/error.
    roi_adata = d2c_adata[d2c_adata.obs["ROI"] == roi]
    safe_roi_name = str(roi).replace(" ", "_").replace("/", "-")

    sq.pl.spatial_scatter(
        roi_adata,
        shape=None,  # no image/library background, just plot coordinates
        color=drugs_of_interest,
        wspace=0.4,
        size=1,
        figsize=(6, 6),
        ncols=3,
        cmap=cmap,
        save=f"drug2cell_spatial_scatter_{safe_roi_name}.pdf",
    )

    # Close all open figures to release matplotlib's memory
    plt.close("all")

    # Delete the subset and force garbage collection
    del roi_adata
    gc.collect()
