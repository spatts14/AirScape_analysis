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

# Colorblind-friendly muted palette
# Based on Okabe-Ito hue positions, desaturated and shifted for scientific figures

condition_palettes = {
    "IPF":   "#5E8FA3",  # muted blue
    "PM08":  "#C9933A",  # muted orange
    "COPD":  "#5E9E7E",  # muted teal-green
    "MICA":  "#9B6B5A",  # muted vermillion
}

diagnosis_palette = {
    "IPF":          "#5E8FA3",  # muted blue
    "LUNG_CANCER":  "#C9933A",  # muted orange
    "COPD":         "#5E9E7E",  # muted teal-green
    "HEALTHY":      "#7A9E6E",  # muted green
    "NO_CRD":       "#8B7BAF",  # muted violet
}

study_palette = {
    "RBH":        "#5E8FA3",  # muted blue
    "PM08":       "#C9933A",  # muted orange
    "REJUVENAIR": "#5E9E7E",  # muted teal-green
    "MICA_III":   "#9B6B5A",  # muted vermillion
}

treatment_arm_palette = {
    "SHAM":      "#7B9EB5",  # soft blue
    "TREATMENT": "#A07B9E",  # soft mauve
}

time_point_palette = {
    "baseline":  "#4A7C99",  # blue
    "6_weeks":   "#B5783A",  # amber-brown
    "6_months":  "#7A9E6E",  # green
}

sex_palette = {
    "male":   "#4A7C99",  # blue
    "female": "#B8836A",  # warm terracotta
}

lung_location_palette = {
    "PROXIMAL": "#4A7C99",  # blue
    "DISTAL":   "#B5783A",  # amber-brown
}

biopsy_type_palette = {
    "CRYOBIOPSY":            "#4A7C99",  # blue
    "TRANSBRONCHIAL BIOPSY": "#B5783A",  # amber-brown
    "RESECTION":             "#7A9E6E",  # green
    "ENDOBRONCHIAL BIOPSY":  "#8B7BAF",  # violet
}

smoking_status_palette = {
    "NEVER":          "#5E9E7E",  # teal-green
    "EX_SMOKER":      "#C9B46A",  # muted yellow-ochre
    "CURRENT_SMOKER": "#9B6B5A",  # muted vermillion
    "UNKNOWN":        "#8A8A85",  # neutral grey
}

# cell type palette
level_2_palette_V2 = {
    # Epithelial
    "Ciliated cells":             "#2E86AB",
    "Goblet cells":               "#52B788",
    "Basal cells":                "#1B4332",
    "Proliferating Basal cells":  "#74C69D",
    "Secretory epithelial cells": "#95D5B2",
    "AT1 cells":                  "#40916C",
    "AT2 cells":                  "#081C15",
    "Proliferating AT2 cells":    "#D8F3DC",
    # Fibroblasts
    "Adventitial fibroblasts":    "#E07A5F",
    "CTHRC1+ fibroblasts":        "#A44A3F",
    "Alveolar fibroblasts":       "#F2CC8F",
    "Lipo-fibroblasts":           "#C47B32",
    # Endothelial
    "Lymphatic endothelial cells":               "#9B5DE5",
    "Pulmonary vein endothelial cell":           "#C77DFF",
    "Blood endothelial cells - unclassified":    "#5A189A",
    "Capillary endothelial cells":               "#E0AAFF",
    "Pulmonary artery endothelial cell":         "#7B2FBE",
    "Pericytes":                                 "#3C096C",
    # Macrophages & monocytes
    "Macrophages":                    "#D62828",
    "Lipid-associated macrophages":   "#F77F00",
    "Interstitial Macrophages":       "#8B1A1A",
    "Airway/Alveolar macrophages":    "#FCBF49",
    "Monocytes":                      "#EAE2B7",
    # T cells & NK cells
    "T cells":      "#006466",
    "CD4+ T cells": "#0081A7",
    "CD8+ T cells": "#00AFB9",
    "NK cells":     "#649889",
    # Other immune
    "B cells":         "#3A86FF",
    "Plasma cells":    "#023E8A",
    "Dendritic cells": "#FF9E00",
    "Mast cells":      "#9D4EDD",
    "Neutrophils":     "#CAFFBF",
    # Other
    "SMC":        "#6D6875",
    "Aerocytes":  "#B5838D",
    "Unknown":    "#ADB5BD",
}
