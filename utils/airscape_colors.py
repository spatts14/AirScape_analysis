"""Color palettes for plotting."""

import seaborn as sns

# Best color blind friendly palette for categorical variables
cat_palette = sns.color_palette("colorblind")

okabe_ito = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# Set color palettes
condition_palettes = {
    "IPF": "#56B4E9",
    "PM08": "#E69F00",
    "COPD": "#009E73",
    "MICA": "#D55E00",
}

diagnosis_palette = {
    "IPF": "#56B4E9",
    "LUNG_CANCER": "#E69F00",
    "COPD": "#009E73",
    "HEALTHY": "#D55E00",
    "NO_CRD": "#CC79A7",

}
study_palette = {
    "RBH": "#56B4E9",
    "PM08": "#E69F00",
    "REJUVENAIR": "#009E73",
    "MICA_III": "#D55E00",
}

treatment_arm_palette = {
    "SHAM": "#56B4E9",
    "TREATMENT": "#CC79A7",
}

time_point_palette = {
    "baseline": "#0072B2",
    "6_weeks": "#D55E00",
    "6_months": "#F0E442",
}

sex_palette = {
    "male": "#0072B2",
    "female": "#E69F00",
}

lung_location_palette = {
    "PROXIMAL": "#0072B2",
    "DISTAL": "#D55E00",
}

biopsy_type_palette = {
    "CRYOBIOPSY": "#0072B2",
    "TRANSBRONCHIAL BIOPSY": "#D55E00",
    "RESECTION": "#F0E442",
    "ENDOBRONCHIAL BIOPSY": "#E69F00"
}

smoking_status_palette = {
    "NEVER": "#009E73",
    "EX_SMOKER": "#F0E442",
    "CURRENT_SMOKER": "#D55E00",
    "UNKNOWN": "#72706C",
}
