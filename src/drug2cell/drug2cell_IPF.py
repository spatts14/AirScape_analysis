"""Drug2Cell for IPF data."""

# Load packages
import gc
import pickle
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
output = dir / "drug2cell" / "output_chembl37"
output.mkdir(exist_ok=True, parents=True)
sc.settings.figdir = output  # set figure directory

# Set colors
cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)

# Variables
drugs_of_interest = [
    "CHEMBL1256391|PIRFENIDONE",
    "CHEMBL3039504|NINTEDANIB ESYLATE",
    "CHEMBL93|ZILEUTON",
]


# Load the custom ChEMBL 37 drug-target dictionary built by parse_database.py
chembl37_dict_path = dir / "drug2cell/database/chembl_37/chembl_37_drug_dictionary.pkl"
with open(chembl37_dict_path, "rb") as f:
    chembl37_dict = pickle.load(f)

# Load adata
adata = ad.read_zarr(dir / "AIRSCAPE/adata_final_object/adata_with_metadata.zarr")

# Subset to IPF and PM08 samples
adata = adata[adata.obs["condition"].isin(["IPF", "PM08"])]

# Calculate drug2cell scores
d2c.score(adata, targets=chembl37_dict, nested=True, use_raw=True)


# Save drug2cell-scored object to adata.uns
d2c_adata = adata.uns["drug2cell"]
del adata
gc.collect()

# Subset to respiratory drugs
plot_args = d2c.util.prepare_plot_args(d2c_adata, categories=["R"])

# Export drug2cell scores to Excel
drugs_present = d2c_adata.var
drugs_present.reset_index().to_excel(output / "drug2cell_scores.xlsx", index=True)

level_cols = ["level_1", "level_2", "level_3"]

for level in level_cols:
    # Make directory for each level
    level_dir = output / level
    level_dir.mkdir(exist_ok=True, parents=True)

    # Set scanpy's figure directory to the level-specific directory
    sc.settings.figdir = level_dir

    # Calculate differentially expressed drug2cell scores for each cell type
    sc.tl.rank_genes_groups(
        d2c_adata,
        method="wilcoxon",
        groupby=level,
        key_added=f"d2c_rank_genes_groups_{level}",
    )

    # Plot differentially expressed drug2cell scores for each cell type
    sc.pl.rank_genes_groups_dotplot(
        d2c_adata,
        key=f"d2c_rank_genes_groups_{level}",
        swap_axes=True,
        dendrogram=False,
        n_genes=5,
        cmap=cmap,
        save=f"drug2cell_rank_genes_groups_dotplot_{level}.pdf",
    )

    # Dotplot for each cell type
    top_drugs_by_celltype_level = plot_top_drugs_by_condition_for_all_celltypes(
        d2c_adata,
        level_col=level,
        rank_key=f"d2c_rank_genes_groups_{level}",
        output_dir=output,
        group_order=["PM08", "IPF"],
    )

    # Respiratory drugs of interest
    sc.pl.dotplot(
        d2c_adata,
        groupby=level,
        swap_axes=True,
        **plot_args,
        cmap=cmap,
        save=f"dotplot_{level}_respiratory.pdf",
    )

# Iterate through each drug in the drugs_of_interest list
for drug in drugs_of_interest:
    # Check if the drug is present in the d2c_adata.var_names
    if drug not in d2c_adata.var_names:
        print(f"Warning: {drug} not found in d2c_adata.var_names. Skipping.")
        continue

    # make directory for each drug
    drug_dir = output / drug.replace("|", "_")
    drug_dir.mkdir(exist_ok=True, parents=True)

    # save plots to the drug-specific directory
    sc.settings.figdir = drug_dir

    # Plot UMAP for the drug across all cell types
    sc.pl.umap(
        d2c_adata,
        color=[drug],
        color_map=cmap,
        save="drug2cell_umap.png",
    )

    for level in level_cols:
        sc.pl.dotplot(
            d2c_adata,
            var_names=[drug],
            groupby=level,
            color_map=cmap,
            save=f"_{level}.pdf",
        )

        # Filter out cells missing either grouping column before the combined dotplot
        mask = d2c_adata.obs[level].notna() & d2c_adata.obs["condition"].notna()
        n_dropped = (~mask).sum()
        if n_dropped > 0:
            print(
                f"Dropping {n_dropped} cells with missing '{level}' or "
                f"'condition' before combined dotplot."
            )
        subset_for_combined = d2c_adata[mask]

        sc.pl.dotplot(
            subset_for_combined,
            var_names=[drug],
            groupby=[level, "condition"],
            standard_scale="var",
            cmap=cmap,
            save=f"_{level}_condition.pdf",
        )

    # Spatial plots
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

        # Plot spatial scatter for the current drug in the current ROI
        sq.pl.spatial_scatter(
            roi_adata,
            shape=None,  # no image/library background, just plot coordinates
            color=[drug],
            wspace=0.4,
            size=0.5,
            figsize=(6, 6),
            ncols=3,
            cmap=cmap,
            vmax=4,
            save=f"{safe_roi_name}_drug2cell_spatial_scatter.png",
        )

        # Close all open figures to release matplotlib's memory
        plt.close("all")

        # Delete the subset and force garbage collection
        del roi_adata
        gc.collect()
