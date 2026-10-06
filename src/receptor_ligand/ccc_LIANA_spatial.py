import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)


def count_matrix(df, cond):
    sub = df[(df[CONDITION_KEY] == cond) & df["sig"]]
    m = pd.crosstab(sub["source"], sub["target"])
    return m


def pretty(pair):
    return pair.replace("^", ARROW)


def top_pairs(df, ascending, n):
    d = df.sort_values("interaction_stat", ascending=ascending)
    d = d.drop_duplicates(["ligand_complex", "receptor_complex"])
    return d.head(n)


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
TOP_N = 20  # number of interactions to show in dotplots (overview and per source)
SIG_RANK = 0.05  # threshold for "significant" interactions
DOTPLOT_PVAL = 0.05  # threshold for dotplots by source (A6)
DOTPLOT_TOP_N = 20  # max number of interactions to show in dotplots by source (A6)


OUTDIR.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = OUTDIR
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

# Sequential colour map for non-negative values (counts, lr_means, local scores).
# Signed values (differences, stats) use the diverging "RdBu_r".
cmap = sns.color_palette("Blues", as_cmap=True)


print("\n[Part D] Spatial L-R maps ...")
sp_dir = OUTDIR / "D_spatial"
sp_dir.mkdir(exist_ok=True)
COND_COLORS = {TEST: "#c0392b", REFERENCE: "#2c6fbb"}
ARROW = " → "
