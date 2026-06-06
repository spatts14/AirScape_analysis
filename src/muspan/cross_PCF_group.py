"""Plot  cPCF group results and run stats on them."""

import json
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set figure aesthetics
sns.set_theme(style="white")
plt.rcParams["figure.figsize"] = (10, 8)
cmap_Blues = sns.color_palette("Blues", as_cmap=True)


def add_metadata(
    metadata_path=Config.METADATA_FILE,
    metadata_sheet_name=Config.SHEET_NAME,
    slide_id=Config.SLIDE_ID_COL,
    data_dict_path=Config.CPCF_JSON_FILE,
):
    """Load experimental results & merge with metadata.

    This function loads a metadata table from an Excel file and a JSON file
    containing computational results (e.g., pair correlation function values
    and confidence intervals). Metadata are joined to the JSON data using
    the `slide_id` column. The output is a tidy DataFrame where each row
    corresponds to one sample and distance value.

    Args:
        metadata_path (str, optional): Path to the Excel metadata file.
            Defaults to `Config.METADATA_FILE`.
        metadata_sheet_name (str, optional): Name of the sheet in the Excel
            file containing metadata. Defaults to `Config.SHEET_NAME`.
        slide_id (str, optional): Column name in the metadata file to use as
            a unique sample identifier (e.g., image ID).
            Defaults to `Config.SLIDE_ID_COL`.
        data_dict_path (str, optional): Path to the JSON file containing
            computational results. Defaults to `Config.JSON_FILE`.

    Returns:
        pandas.DataFrame: A tidy DataFrame with columns:
            - `sample` (str): Sample identifier.
            - `r` (float): Distance values.
            - `PCF` (float): Pair correlation function values.
            - `ci_low` (float): Lower confidence interval.
            - `ci_high` (float): Upper confidence interval.
            - Plus any additional metadata columns from the Excel file.

    Notes:
        If a sample is present in the JSON data but missing from the
        metadata, only the JSON-derived values will appear for that row.
    """
    # Load metadata
    df = pd.read_excel(
        metadata_path, sheet_name=metadata_sheet_name
    )  # not working with the sheet name config file

    # set index to image_ID so it becomes the key
    df = df.set_index(slide_id)

    # convert to dict of dicts
    metadata_dict = df.to_dict(orient="index")

    # load your matrices
    with open(data_dict_path) as f:
        cPCF_data = json.load(f)

    # flatten into tidy dataframe
    records = []
    for sample, vals in cPCF_data.items():
        for i, r in enumerate(vals["r"]):
            rec = {
                "sample": sample,
                "r": r,
                "PCF": vals["PCF"][i],
                "ci_low": vals["confidence_intervals"][0][i],
                "ci_high": vals["confidence_intervals"][1][i],
                **metadata_dict.get(sample, {}),  # merge metadata
            }
            records.append(rec)

    df_complete = pd.DataFrame(records)
    return df_complete


def main():
    """Main execution function."""
    try:
        # Make directories if they don't exist
        logging.info("Creating output directories...")
        Config.CPCF_COMPARISON_PATH.mkdir(parents=True, exist_ok=True)

        # Set saving population names
        save_pop_A = (
            Config.POP_A.replace("/", "_").replace(" ", "_").replace("+", "pos")
        )
        save_pop_B = (
            Config.POP_B.replace("/", "_").replace(" ", "_").replace("+", "pos")
        )

        # Load all saved MuSpAn domains
        logging.info("Loading saved data...")

        # Add metadata to cPCF
        df_complete = add_metadata(
            metadata_path=Config.METADATA_FILE,
            metadata_sheet_name=Config.SHEET_NAME,
            data_dict_path=Config.CPCF_JSON_FILE,
        )

        logging.info(f"Plotting {Config.POP_A} and {Config.POP_B} individually...")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.lineplot(
            data=df_complete,
            x="r",
            y="PCF",
            hue=Config.GROUPBY,
            units="sample",
            estimator=None,
            alpha=0.8,
            ax=ax,
        )

        # Add per-sample confidence bands
        for sample, sub in df_complete.groupby("sample"):
            plt.fill_between(
                sub["r"],
                sub["ci_low"],
                sub["ci_high"],
                alpha=0.2,
                color=sns.color_palette("tab10")[
                    list(df_complete["treatment"].unique()).index(
                        sub["treatment"].iloc[0]
                    )
                ],
            )

        plt.title(f"{Config.POP_A} and {Config.POP_B} cPCF by {Config.GROUPBY}")
        plt.ylabel("cPCF")
        plt.xlabel("Distance (µm)")
        plt.savefig(
            Config.CPCF_COMPARISON_PATH
            / f"{save_pop_A}_and_{save_pop_B}_cPCF_individual_samples.png",
            bbox_inches="tight",
            facecolor="white",
            dpi=300,
        )

        logging.info(f"Plotting {Config.POP_A} and {Config.POP_B} aggregate...")
        # Mean with 95% confidence interval
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.lineplot(
            data=df_complete,
            x="r",
            y="PCF",
            hue=Config.GROUPBY,
            # Remove units and estimator=None to allow aggregation
            errorbar=("ci", 95),  # 95% confidence interval
            ax=ax2,
        )

        plt.title(
            f"{Config.POP_A} and {Config.POP_B} cPCF by {Config.GROUPBY} (mean ± 95% CI)"  # noqa: E501
        )
        plt.ylabel("cPCF")
        plt.xlabel("Distance (µm)")
        plt.savefig(
            Config.CPCF_COMPARISON_PATH
            / f"{save_pop_A}_and_{save_pop_B}_cPCF_aggregated.png",
            bbox_inches="tight",
            facecolor="white",
            dpi=300,
        )

        # Print final summary
        logging.info("=== ANALYSIS COMPLETE ===")
        logging.info(f"Results saved to: {Config.CPCF_COMPARISON_PATH}")

    except Exception as e:
        logging.error(f"Script execution failed: {e!s}")
        raise


if __name__ == "__main__":
    main()