import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import muspan as ms

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

base_dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
)

domains_dir = base_dir / "output" / "muspan" / "domains"
docs_dir = base_dir / "docs" / "xenium_explorer_cell_IDs"

# Set 1
pooled_domain_path = domains_dir / "MICA_III_319_315_311_muspan_domain.muspan"
DONOR_CSVS = {
    "MICA_III_311": docs_dir / "MICA_III_311_cells_stats.csv",
    "MICA_III_315": docs_dir / "MICA_III_315_cells_stats.csv",
    "MICA_III_319": docs_dir / "MICA_III_319_cells_stats.csv",
}

# # Set 2
# pooled_domain_path = domains_dir / "MICA_III_325_337_379_muspan_domain.muspan"  # pooled domain file
# DONOR_CSVS = {
#     "MICA_III_325": docs_dir / "MICA_III_325_cells_stats.csv",
#     "MICA_III_337": docs_dir / "MICA_III_337_cells_stats.csv",
#     "MICA_III_379": docs_dir / "MICA_III_379_cells_stats.csv",
# }

for name, path in DONOR_CSVS.items():
    print(name, "->", path, "exists:", path.exists())
print("pooled domain exists:", pooled_domain_path.exists())

pooled_domain = ms.io.load_domain(str(pooled_domain_path))

print(pooled_domain)
print("n_objects:", pooled_domain.n_objects)
print("labels present:", list(pooled_domain.labels.keys()))

for donor_name, path in DONOR_CSVS.items():
    print(f"--- {donor_name} ({path.name}) ---")
    with open(path) as f:
        for i, line in zip(range(4), f):
            print(i, repr(line))
    print()

# print head for each donor CSV to check column names
for donor_name, path in DONOR_CSVS.items():
    print(f"--- {donor_name} ({path.name}) ---")
    df = pd.read_csv(
        path, skiprows=2
    )  # skip the first two rows of the CSV, which are not part of the data
    print(df.head())
    print()

    # Edit this if the column name printed above isn't "Cell ID"
CELL_ID_COL = "Cell ID"

# Sanity check: does this column line up with the domain's own Cell ID labels?
domain_cell_ids_sample = {
    str(c) for c in pooled_domain.labels["Cell ID"]["labels"][:20]
}
csv_cell_ids_sample = set(df[CELL_ID_COL].astype(str)[:20])
print("Domain Cell ID examples:", list(domain_cell_ids_sample)[:5])
print("CSV cell ID examples:   ", list(csv_cell_ids_sample)[:5])


def filter_domain_to_donor(
    pooled_domain_path, donor_cell_ids_csv, cell_id_col=CELL_ID_COL
):
    """Return a fresh domain containing only the cells listed in donor_cell_ids_csv."""
    domain = ms.io.load_domain(str(pooled_domain_path))

    donor_df = pd.read_csv(
        donor_cell_ids_csv, skiprows=2
    )  # skip the first two rows of the CSV, which are not part of the data
    donor_cell_ids = set(donor_df[cell_id_col].astype(str))

    domain_cell_ids = [str(c) for c in domain.labels["Cell ID"]["labels"]]
    n_before = domain.n_objects

    membership = [
        "keep" if cid in donor_cell_ids else "drop" for cid in domain_cell_ids
    ]
    domain.add_labels(label_name="_donor_filter", labels=membership)

    query_remove = ms.query.query(domain, ("label", "_donor_filter"), "is", "drop")
    domain.delete_objects(query_remove)

    n_after = domain.n_objects
    print(
        f"{donor_cell_ids_csv.stem}: {n_before} -> {n_after} objects "
        f"({n_before - n_after} removed, {len(donor_cell_ids)} IDs in CSV)"
    )

    return domain


split_domains = {}

for donor_name, csv_path in DONOR_CSVS.items():
    split_domains[donor_name] = filter_domain_to_donor(pooled_domain_path, csv_path)

total_split = sum(d.n_objects for d in split_domains.values())
print()
print(f"Pooled domain objects:      {pooled_domain.n_objects}")
print(f"Sum across split domains:   {total_split}")
if total_split != pooled_domain.n_objects:
    print(
        "WARNING: these don't match — check for cells missing from all CSVs, "
        "or cells that appear in more than one donor's CSV (overlap)."
    )

    fig, axes = plt.subplots(1, len(split_domains), figsize=(3 * len(split_domains), 3))

if len(split_domains) == 1:
    axes = [axes]

for ax, (donor_name, domain) in zip(axes, split_domains.items()):
    # Get (or create) cell centroids to plot
    try:
        ms.query.query(domain, ("Collection",), "is", "Cell centroids")
    except Exception:
        domain.convert_objects(
            population=("Collection", "Cell boundaries"),
            object_type="point",
            conversion_method="centroids",
            collection_name="Cell centroids",
            inherit_collections=False,
        )

    xy = ms.query.query(domain, ("Collection",), "is", "Cell centroids")

    color_by = ("label", "Cell Type") if "Cell Type" in domain.labels else None

    plt.sca(ax)
    ms.visualise.visualise(
        domain,
        color_by=color_by,
        objects_to_plot=xy,
        marker_size=0.5,
        add_scalebar=True,
        scalebar_kwargs={
            "size": 500,
            "label": "500\u00b5m",
            "loc": "lower right",
            "pad": 0.1,
            "color": "black",
            "frameon": False,
            "size_vertical": 2,
        },
    )
    ax.set_title(f"{donor_name} (n={domain.n_objects})")

plt.tight_layout()
plt.show()

for donor_name, domain in split_domains.items():
    # Rename domain
    domain.name = f"{donor_name}"

    # Save the domain to disk
    ms.io.save_domain(
        domain,
        name_of_file=f"{donor_name}_muspan_domain",
        path_to_save=str(domains_dir),
    )
    print(
        f"Saved {donor_name} with domain name {domain.name} -> {domains_dir / (donor_name + '_muspan_domain.muspan')}"
    )
