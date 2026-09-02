"""Build the ChEMBL 37 drug-target dictionary for use with drug2cell.

Parses the ChEMBL 37 SQLite database, annotates target classes,
filters activities, and saves the final drugs:targets dictionary
ready to pass to d2c.score(adata, targets=..., nested=True).

Requires four manually-downloaded reference files (see notes at bottom):
    IDG_TargetList_Y4.csv
    HGNC_GID177_Ion-channels.txt
    HGNC_GID139_G-protein-coupled-receptors.txt
    HGNC_GID71_Nuclear-hormone-receptors.txt
"""

import pickle
import sqlite3
from pathlib import Path

import pandas as pd

import drug2cell as d2c

# Paths
dir = Path(
    "/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium"
)
ref_dir = dir / "output/drug2cell/database/chembl_37"
CHEMBL_DB_PATH = ref_dir / "chembl_37_sqlite/chembl_37.db"

# Connect to ChEMBL database
con = sqlite3.connect(CHEMBL_DB_PATH)

# Log actual schema for the tables we depend on, so any future schema drift
# is visible in the logs rather than causing a bare KeyError partway through
for table in [
    "activities",
    "assays",
    "target_dictionary",
    "target_components",
    "component_synonyms",
    "drug_mechanism",
    "molecule_dictionary",
    "molecule_atc_classification",
    "atc_classification",
]:
    cols = [row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()]

# Load core tables
activities = pd.read_sql_query("SELECT * from activities", con)
activities.columns = [f"activities|{x}" for x in activities.columns]

assays = pd.read_sql_query("SELECT * from assays", con)
assays.columns = [f"assays|{x}" for x in assays.columns]

target_dictionary = pd.read_sql_query("SELECT * from target_dictionary", con)
target_dictionary.columns = [
    f"target_dictionary|{x}" for x in target_dictionary.columns
]


target_components = pd.read_sql_query("SELECT * from target_components", con)
target_components.columns = [
    f"target_components|{x}" for x in target_components.columns
]

component_synonyms = pd.read_sql_query("SELECT * from component_synonyms", con)
component_synonyms.columns = [
    f"component_synonyms|{x}" for x in component_synonyms.columns
]

drug_mechanism = pd.read_sql_query("SELECT * from drug_mechanism", con)
drug_mechanism.columns = [f"drug_mechanism|{x}" for x in drug_mechanism.columns]

molecule_dictionary = pd.read_sql_query("SELECT * from molecule_dictionary", con)
molecule_dictionary.columns = [
    f"molecule_dictionary|{x}" for x in molecule_dictionary.columns
]

molecule_atc_classification = pd.read_sql_query(
    "SELECT * from molecule_atc_classification", con
)
molecule_atc_classification.columns = [
    f"molecule_atc_classification|{x}" for x in molecule_atc_classification.columns
]

atc_classification = pd.read_sql_query("SELECT * from atc_classification", con)
atc_classification.columns = [
    f"atc_classification|{x}" for x in atc_classification.columns
]

# Merge activities + assays
final_df = activities.merge(
    assays, how="left", left_on="activities|assay_id", right_on="assays|assay_id"
)

# assays|curated_by was present in ChEMBL 30 but is not guaranteed to exist
# in later releases (confirmed missing in 37). Build the selection list
# dynamically so this doesn't hard-crash on future schema changes either.
desired_columns = [
    "activities|activity_id",
    "activities|assay_id",
    "activities|molregno",
    "activities|pchembl_value",
    "activities|type",
    "activities|standard_relation",
    "activities|standard_value",
    "activities|standard_units",
    "activities|standard_flag",
    "activities|standard_type",
    "activities|activity_comment",
    "assays|description",
    "assays|assay_type",
    "assays|tid",
    "assays|confidence_score",
    "assays|curated_by",
    "assays|chembl_id",
]
available_columns = [c for c in desired_columns if c in final_df.columns]
missing_columns = [c for c in desired_columns if c not in final_df.columns]

final_df = final_df[available_columns]

# Merge with drug_mechanism
final_df = final_df.merge(
    drug_mechanism[
        [
            "drug_mechanism|molregno",
            "drug_mechanism|mechanism_of_action",
            "drug_mechanism|tid",
            "drug_mechanism|action_type",
        ]
    ],
    how="outer",
    left_on=["activities|molregno", "assays|tid"],
    right_on=["drug_mechanism|molregno", "drug_mechanism|tid"],
)

# Reconcile molregno/tid columns
ind = final_df[
    (final_df["drug_mechanism|molregno"] == final_df["drug_mechanism|molregno"])
    & (final_df["activities|molregno"] != final_df["activities|molregno"])
].index
final_df["activities_drug_mechanism|molregno"] = final_df["activities|molregno"].copy()
final_df.loc[ind, "activities_drug_mechanism|molregno"] = final_df.loc[
    ind, "drug_mechanism|molregno"
]
del ind

ind = final_df[
    (final_df["drug_mechanism|tid"] == final_df["drug_mechanism|tid"])
    & (final_df["assays|tid"] != final_df["assays|tid"])
].index
final_df["assays_drug_mechanism|tid"] = final_df["assays|tid"].copy()
final_df.loc[ind, "assays_drug_mechanism|tid"] = final_df.loc[ind, "drug_mechanism|tid"]
del ind

# Merge compound info
final_df = final_df.merge(
    molecule_dictionary[
        [
            "molecule_dictionary|molregno",
            "molecule_dictionary|pref_name",
            "molecule_dictionary|chembl_id",
            "molecule_dictionary|max_phase",
            "molecule_dictionary|molecule_type",
            "molecule_dictionary|oral",
            "molecule_dictionary|parenteral",
            "molecule_dictionary|topical",
            "molecule_dictionary|black_box_warning",
            "molecule_dictionary|natural_product",
        ]
    ],
    how="left",
    left_on="activities_drug_mechanism|molregno",
    right_on="molecule_dictionary|molregno",
)

final_df = final_df.merge(
    molecule_atc_classification[
        ["molecule_atc_classification|molregno", "molecule_atc_classification|level5"]
    ],
    how="left",
    left_on="activities_drug_mechanism|molregno",
    right_on="molecule_atc_classification|molregno",
)

final_df = final_df.merge(
    atc_classification[
        [
            "atc_classification|level1",
            "atc_classification|level2",
            "atc_classification|level3",
            "atc_classification|level4",
            "atc_classification|level5",
            "atc_classification|level1_description",
            "atc_classification|level2_description",
            "atc_classification|level3_description",
            "atc_classification|level4_description",
            "atc_classification|who_name",
        ]
    ],
    how="left",
    left_on="molecule_atc_classification|level5",
    right_on="atc_classification|level5",
)

# Merge target info (GENE_SYMBOL only)
targets_final = target_dictionary.merge(
    target_components,
    how="left",
    left_on="target_dictionary|tid",
    right_on="target_components|tid",
)
targets_final = targets_final.merge(
    component_synonyms,
    how="left",
    left_on="target_components|component_id",
    right_on="component_synonyms|component_id",
)
targets_final = targets_final[
    targets_final["component_synonyms|syn_type"] == "GENE_SYMBOL"
]

final_df = final_df.merge(
    targets_final[
        [
            "target_dictionary|tid",
            "target_dictionary|target_type",
            "target_dictionary|pref_name",
            "target_dictionary|organism",
            "target_dictionary|chembl_id",
            "component_synonyms|component_synonym",
            "component_synonyms|syn_type",
        ]
    ],
    how="left",
    left_on="assays_drug_mechanism|tid",
    right_on="target_dictionary|tid",
)

# Restrict to human targets
final_df = final_df[final_df["target_dictionary|organism"] == "Homo sapiens"]

# Add target class (Kinase, GPCR, Ion Channel, NHR)
idg = pd.read_csv(ref_dir / "IDG_TargetList_Y4.csv")

targetclass_dict = {}
for c in set(idg["IDGFamily"]):
    targetclass_dict[c] = list(idg[idg["IDGFamily"] == c]["Gene"])

ion = pd.read_csv(ref_dir / "HGNC_GID177_Ion-channels.txt", sep="\t")
gpcr = pd.read_csv(ref_dir / "HGNC_GID139_G-protein-coupled-receptors.txt", sep="\t")
nr = pd.read_csv(ref_dir / "HGNC_GID71_Nuclear-hormone-receptors.txt", sep="\t")

targetclass_dict["Ion Channel"] = list(
    set(targetclass_dict.get("Ion Channel", []) + list(ion["Approved symbol"]))
)
targetclass_dict["GPCR"] = list(
    set(targetclass_dict.get("GPCR", []) + list(gpcr["Approved symbol"]))
)
targetclass_dict["NHR"] = list(nr["Approved symbol"].unique())


def which_class(dictionary, value):
    """Map a gene symbol to its target class(es)."""
    out = "none"
    for k in dictionary.keys():
        if value in dictionary[k]:
            out = k if out == "none" else f"{out};{k}"
    return out


final_df["target_class"] = final_df["component_synonyms|component_synonym"].copy()
final_df["target_class"] = [
    which_class(targetclass_dict, t) for t in final_df["target_class"]
]

# Filter activities and build the drugs:targets dictionary
thresholds_dict = {
    "none": 6,  # 1 uM
    "NHR": 7,  # 100 nM
    "GPCR": 7,  # 100 nM
    "Ion Channel": 5,  # 10 uM
    "Kinase": 7.53,  # 30 nM
}

filtered_df = d2c.chembl.filter_activities(
    dataframe=final_df,
    drug_max_phase=1,  # look at all drugs including preclinical
    assay_type="F",
    add_drug_mechanism=True,
    remove_inactive=True,
    include_active=True,
    pchembl_target_column="target_class",
    pchembl_threshold=thresholds_dict,
)

chembldict = d2c.chembl.create_drug_dictionary(
    filtered_df,
    drug_grouping="ATC_level",
)

# Save everything
final_df.to_pickle(ref_dir / "chembl_37_merged_genesymbols_humans_ALL.pkl")
with open(ref_dir / "chembl_37_drug_dictionary_ALL.pkl", "wb") as f:
    pickle.dump(chembldict, f)
