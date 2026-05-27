"""Add cell annotations from CSV files to AnnData object and visualize with UMAP."""

import os
from pathlib import Path

import pandas as pd
import scanpy as sc


def collect_all_annotations(base_dir, level: str):
    """Traverse directory structure to find all *_annotations.csv files.

    Args:
        base_dir: Base directory containing subdirectories with annotation files
        level: Annotation level to filter files (e.g., "level_2")

    Returns:
        pd.DataFrame: Combined annotations from all CSV files
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    all_annotations = []
    pattern = f"*_{level}_*_annotations.csv"

    # Find all subdirectories in base_dir
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            # Look for 'files' directory within each subdirectory
            files_dir = subdir / "files"

            if files_dir.exists() and files_dir.is_dir():
                # Find all *_annotations.csv files
                annotation_files = list(files_dir.glob(pattern))

                for csv_file in annotation_files:
                    print(f"Found: {csv_file}")

                    try:
                        # Read the CSV file
                        df = pd.read_csv(csv_file)

                        # Add source information
                        df["source_subdir"] = subdir.name
                        df["source_file"] = csv_file.name

                        all_annotations.append(df)
                        print(f"Loaded {len(df)} annotations from {subdir.name}")

                    except Exception as e:
                        print(f"Error reading {csv_file}: {e}")

    if not all_annotations:
        raise ValueError(f"No files matching '{pattern}' were found in {base_dir}")

    # Combine all annotations
    combined = pd.concat(all_annotations, ignore_index=True)

    # Add column for annotation level
    combined["annotation_level"] = level

    # Confirm correct columns are present
    expected_columns = {"cell_annotation", "cell_id", "annotation_level"}
    missing_columns = expected_columns - set(combined.columns)
    if missing_columns:
        raise ValueError(
            f"Missing expected columns in combined annotations: {missing_columns}"
            f"Available columns: {combined.columns.tolist()}"
        )

    # Check column names
    print(f"\nTotal annotations loaded: {len(combined)}")
    print(f"Columns: {combined.columns.tolist()}")

    return combined


def rename_cells_from_annotations(
    adata,
    annotations_df,
    level: str,
    cell_id_col: str = "cell_id",
    annotation_col: str = "cell_annotation",
):
    """Rename cells in AnnData object using annotations from CSV.

    Args:
        adata: AnnData object
        annotations_df: DataFrame with cell IDs and annotations
        level: Annotation level (used for naming new column)
        cell_id_col: Column name in annotations_df containing cell IDs
        annotation_col: Column name in annotations_df containing annotations

    Returns:
        AnnData object with updated cell annotations
    """
    # Check if columns exist
    if cell_id_col not in annotations_df.columns:
        raise ValueError(
            f"Column '{cell_id_col}' not found in annotations."
            f"Available: {annotations_df.columns.tolist()}"
        )

    if annotation_col not in annotations_df.columns:
        raise ValueError(
            f"Column '{annotation_col}' not found in annotations."
            f"Available: {annotations_df.columns.tolist()}"
        )

    # Check if cell_id exists in adata.obs
    if "cell_id" not in adata.obs.columns:
        print(
            "Warning: 'cell_id' not found in adata.obs, using adata.obs_names instead"
        )

    # Create mapping dictionary
    cell_id_to_annotation = dict(
        zip(annotations_df[cell_id_col], annotations_df[annotation_col])
    )

    print(f"Mapping {len(cell_id_to_annotation)} cell IDs to annotations")

    # Map annotations to adata
    adata.obs[level] = adata.obs["cell_id"].map(cell_id_to_annotation)

    # Check for unmapped cells
    unmapped = adata.obs[level].isna().sum()
    if unmapped > 0:
        print(f"Warning: {unmapped} cells could not be mapped to annotations")
        print(f"Total cells in adata: {len(adata)}")
        print(f"Successfully mapped: {len(adata) - unmapped}")
    else:
        print(f"Successfully mapped all {len(adata)} cells!")

    # Print summary
    print("\nAnnotation summary:")
    print(adata.obs[level].value_counts())

    return adata


# Define base directory
base_dir = Path(os.getenv("BASE_DIR"))

# Set variables
level_list = ["level_2", "level_3"]  # Define annotation levels to process
level_list_str = "_".join(level_list)  # make save safe

# Set paths to files and data
annotation_file_path = Path(os.getenv("CELLTYPE_SUBSET_DIR"))
annotate_path = Path(os.getenv("ANNOTATE_DIR"))
output_path = annotate_path / level_list_str
output_path.mkdir(exist_ok=True)

# set fig directory for saving
sc.settings.figdir = output_path

# Set variables
cell_id_col = "cell_id"
annotation_col = "cell_annotation"

# Load adata
adata = sc.read_h5ad(annotate_path / "adata.h5ad")

# Concatenate annotations for each subset and level
annotations_dict = {}
for level in level_list:
    print(f"Saving concatenated files for {level}")

    # Collect all annotations for the specified level
    annotations_df = collect_all_annotations(annotation_file_path, level=level)

    # Save combined annotations to CSV for reference
    annotations_df.to_csv(
        output_path / f"combined_annotations_{level}.csv", index=False
    )

    # Store annotations for each level
    annotations_dict[level] = annotations_df

for level in level_list:
    print(f"Mapping {level} to full dataset")

    # Add annotations as a new column
    adata = rename_cells_from_annotations(
        adata,
        annotations_dict[level],
        level=level,
        cell_id_col=cell_id_col,
        annotation_col=annotation_col,
    )
    # Debugging checks
    print(f"Total cells: {len(adata)}")
    print(f"Cells with annotations: {adata.obs[level].notna().sum()}")
    print(f"Unique annotations: {adata.obs[level].nunique()}")

# Save updated adata
adata.write_h5ad(annotate_path / f"adata_{level_list_str}.h5ad")

# Visualize UMAP
for level in level_list:
    sc.pl.umap(adata, color=level, save=f"_{level}_umap.png", show=False)

print("Annotation and visualization complete!")
