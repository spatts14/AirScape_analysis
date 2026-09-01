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
    groupby="condition",
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
        groupby : str
            Column to split the dotplot by within each cell type, default "condition".
        n_top : int
            Number of top drugs to show per cell type.
        cmap : colormap
            Colormap for the dotplot.

    Returns:
        dict mapping cell type -> list of top drug names plotted.
    """
    celltypes = d2c_adata.obs[level_col].unique().tolist()
    results = {}

    # Create directory
    output_dir = Path(sc.settings.figdir) / level_col
    output_dir.mkdir(exist_ok=True, parents=True)

    # Loop over cell types and plot top drugs
    for celltype in celltypes:
        print(f"Plotting top {n_top} drugs for '{celltype}'...")

        top_drugs = (
            sc.get.rank_genes_groups_df(d2c_adata, group=celltype, key=rank_key)
            .head(n_top)["names"]
            .tolist()
        )

        if not top_drugs:
            print(f"  Skipping '{celltype}': no ranked drugs found.")
            continue

        mask = d2c_adata.obs[level_col] == celltype
        subset = d2c_adata[mask]

        safe_name = str(celltype).replace(" ", "_").replace("/", "-")

        sc.pl.dotplot(
            subset,
            var_names=top_drugs,
            groupby=groupby,
            cmap=cmap,
            save=f"{safe_name}_d2c_top_drugs_by_{groupby}.pdf",
        )

        results[celltype] = top_drugs

    return results


# Set working directory
dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output"
)

# Set output dir
output = dir / "drug2cell"
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

# Export drug2cell scores to Excel
drugs_present = adata.uns["drug2cell"].var
drugs_present.to_excel(output / "drug2cell_scores.xlsx", index=False)


# Calculate differentially expressed drug2cell scores for each cell type
sc.tl.rank_genes_groups(
    adata.uns["drug2cell"],
    method="wilcoxon",
    groupby="level_2",
    key_added="d2c_rank_genes_groups_level_2",
)

sc.tl.rank_genes_groups(
    adata.uns["drug2cell"],
    method="wilcoxon",
    groupby="level_3",
    key_added="d2c_rank_genes_groups_level_3",
)

# Plot differentially expressed drug2cell scores for each cell type
sc.pl.rank_genes_groups_dotplot(
    adata.uns["drug2cell"],
    key="d2c_rank_genes_groups_level_2",
    swap_axes=True,
    dendrogram=False,
    n_genes=5,
    cmap=cmap,
    save="drug2cell_rank_genes_groups_dotplot_level_2.pdf",
)

sc.pl.rank_genes_groups_dotplot(
    adata.uns["drug2cell"],
    key="d2c_rank_genes_groups_level_3",
    swap_axes=True,
    dendrogram=False,
    n_genes=5,
    cmap=cmap,
    save="drug2cell_rank_genes_groups_dotplot_level_3.pdf",
)

# UMAP
sc.pl.umap(adata.uns["drug2cell"], color=[drugs_of_interest, "level_2"], color_map=cmap)

# Spatial plots
# Confirm the ROI column and spatial coordinates carried over into the drug2cell AnnData
assert "ROI" in adata.obs.columns, "ROI column missing from drug2cell_adata.obs"
assert "spatial" in adata.obsm, "spatial coordinates missing from drug2cell_adata.obsm"

rois = adata.obs["ROI"].unique().tolist()

for roi in rois:
    print(f"Plotting ROI: {roi}")
    roi_adata = adata[adata.obs["ROI"] == roi].copy()
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
