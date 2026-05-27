"""Clean manual metadata to ensure consistency and usability for downstream analyses."""

from pathlib import Path

import pandas as pd

# Set directory
path = Path(
    "/Volumes/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/"
)

metadata = pd.read_excel(
    path / "data/meta/STx_meta_analysis_only.xlsx",
    sheet_name="ROI_metadata",
    index_col="ROI",
)

# Strip column names of leading/trailing whitespace
metadata.columns = metadata.columns.str.strip()

# Clean string columns
for col in metadata.columns:
    if metadata[col].dtype == object:
        metadata[col] = (
            metadata[col]
            .str.strip()  # remove leading/trailing whitespace
            .str.replace(r"\s+", " ", regex=True)  # collapse internal multiple spaces
            .str.upper()  # standardise capitalisation
        )

# Save cleaned metadata as csv file
output_file = path / "data/meta/STx_meta_analysis_only_cleaned.csv"
metadata.to_csv(output_file)
print(f"Cleaned metadata saved to {output_file}")
