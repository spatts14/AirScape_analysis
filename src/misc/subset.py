from pathlib import Path

import anndata as ad

# CONFIG
INPUT_DIR = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output"
)
INPUT_ZARR = INPUT_DIR / "AIRSCAPE/adata_final_object/adata_with_metadata.zarr"
OUTDIR = INPUT_DIR / "ccc" / "IPFvPM08"

CONDITION_KEY = "condition"
CONDITIONS = ["IPF", "PM08"]
REFERENCE = "PM08"  # baseline: red = higher in IPF, blue = lower in IPF

GROUP_KEY = "level_2"  # cell types used as sources/targets
SAMPLE_KEY = "ROI"  # replicate unit for pseudobulk (Part B)
DROP_ROIS = ["PM08_159"]
# Cell types removed before analysis
DROP_CELL_TYPES = [
    "Alveolar fibroblasts (collagen high)",  # only present in one donor
]

# 1. Load and subset
adata = ad.read_zarr(INPUT_ZARR)

# Subset to IPF and PM08 samples
# adata = adata[adata.obs[CONDITION_KEY].isin(CONDITIONS)]
# # Exclude PM08_159
# adata = adata[~adata.obs[SAMPLE_KEY].isin(DROP_ROIS)]
# Remove cell types listed in DROP_CELL_TYPES
adata = adata[~adata.obs[GROUP_KEY].isin(DROP_CELL_TYPES)]

# Keep only 2 IPF and 2 PM08 samples for toy example
# (otherwise the pseudobulk step takes a long time)
adata = adata[
    adata.obs[SAMPLE_KEY].isin(["IPF_RBH_16", "IPF_RBH_01", "PM08_163", "PM08_169"])
]

# Remove cells without a cell-type label (NaN / "nan") - they break the plots
lab = adata.obs[GROUP_KEY]
unlabelled = lab.isna() | lab.astype(str).str.lower().isin(["nan", "none", ""])
print(f"Removing {int(unlabelled.sum())} cells without a {GROUP_KEY} label")
adata = adata[~unlabelled].copy()  # .copy() avoids view warnings

# Save the subsetted object for later use
SUBSET_ZARR = INPUT_DIR / "AIRSCAPE" / "subset_adata" / "toy.zarr"
print(f"Saving subsetted object to {SUBSET_ZARR}")
adata.write_zarr(SUBSET_ZARR)
