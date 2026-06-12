"""Color palettes for plotting."""

#from patsy.mgcv_cubic_splines import cc
import matplotlib.colors as mcolors
#import colorsys
import seaborn as sns

# Predefined color palettes for consistent plotting across modules
condition_palettes = {
    "IPF":   "#6A7FB5",  # slate blue
    "PM08":  "#B07D4A",  # warm tan
    "COPD":  "#7EB0B8",  # dusty teal
    "MICA":  "#A67B8A",  # muted rose
}

diagnosis_palette = {
    "IPF":          "#6A7FB5",  # slate blue
    "LUNG_CANCER":  "#B07D4A",  # warm tan
    "COPD":         "#7EB0B8",  # dusty teal
    "HEALTHY":      "#8EA882",  # sage
    "NO_CRD":       "#A67B8A",  # muted rose
}

study_palette = {
    "RBH":        "#6A7FB5",  # slate blue
    "PM08":       "#B07D4A",  # warm tan
    "REJUVENAIR": "#7EB0B8",  # dusty teal
    "MICA_III":   "#A67B8A",  # muted rose
}

treatment_arm_palette = {
    "SHAM":      "#6A7FB5",  # slate blue
    "TREATMENT": "#B07D4A",  # warm tan
}

time_point_palette = {
    "baseline": "#4A6699",  # deeper blue
    "6_weeks":  "#B07D4A",  # warm tan
    "6_months": "#7EB0B8",  # dusty teal
}

sex_palette = {
    "male":   "#6A7FB5",  # slate blue
    "female": "#B07D4A",  # warm tan
}

lung_location_palette = {
    "PROXIMAL": "#4A6699",  # deeper blue
    "DISTAL":   "#B07D4A",  # warm tan
}

biopsy_type_palette = {
    "CRYOBIOPSY":            "#6A7FB5",  # slate blue
    "TRANSBRONCHIAL BIOPSY": "#B07D4A",  # warm tan
    "RESECTION":             "#7EB0B8",  # dusty teal
    "ENDOBRONCHIAL BIOPSY":  "#A67B8A",  # muted rose
}

smoking_status_palette = {
    "NEVER":          "#8EA882",  # sage
    "EX_SMOKER":      "#C4A85A",  # ochre
    "CURRENT_SMOKER": "#A05A4A",  # muted brick
    "UNKNOWN":        "#8A8A85",  # neutral grey
}

level_1_palette = {
    "Airway epithelial cells":  "#7E9478",
    "Alveolar epithelial cells": "#7A6EA8",
    "Immune cells": "#B8964E",
    "Stromal cells": "#89B9C4",
    "Endothelial cells": "#A05A4A",
    "Unknown": "#828383",
}

level_2_palette= {
    # Epithelial
    "Ciliated cells":             "#5B8FA8",
    "Goblet cells":               "#7EB5A6",
    "Basal cells":                "#4A7B6F",
    "Proliferating Basal cells":  "#3D6B5E",
    "Secretory epithelial cells": "#89C4B0",
    "AT1 cells":                  "#A8C5A0",
    "AT2 cells":                  "#6B9E78",
    "Proliferating AT2 cells":    "#4E7D5B",
    # Fibroblasts
    "Adventitial fibroblasts":    "#C4956A",
    "CTHRC1+ fibroblasts":        "#B07D50",
    "Alveolar fibroblasts":       "#D4A97A",
    "Lipo-fibroblasts":           "#E2C49A",
    # Endothelial
    "Lymphatic endothelial cells":               "#8B7CB3",
    "Pulmonary vein endothelial cell":           "#A08DB8",
    "Blood endothelial cells - unclassified":    "#6B5E9E",
    "Capillary endothelial cells":               "#B8A9CC",
    "Pulmonary artery endothelial cell":         "#7A6EA8",
    "Pericytes":                                 "#C5B8D6",
    # Macrophages & monocytes
    "Macrophages":                    "#C17B6E",
    "Lipid-associated macrophages":   "#B56355",
    "Interstitial Macrophages":       "#D4957F",
    "Airway/Alveolar macrophages":    "#A05A4A",
    "Monocytes":                      "#E8B4A0",
    # T cells & NK cells
    "T cells":      "#7E9E6E",
    "CD4+ T cells": "#92B080",
    "CD8+ T cells": "#5E7E50",
    "NK cells":     "#B5C99A",
    # Other immune
    "B cells":         "#6E8FAA",
    "Plasma cells":    "#5A7A9A",
    "Dendritic cells": "#B8A06E",
    "Mast cells":      "#9E8B5A",
    "Neutrophils":     "#D4C07A",
    # Other
    "SMC":      "#9E9E8A",
    "Aerocytes": "#7A9E9A",
    "Unknown":  "#A0A0A0",
}

# Convert to ListedColormap for use in scanpy/squidpy
level_2_listed = mcolors.ListedColormap([level_2_palette[key] for key in level_2_palette.keys()])


# # Define a function to filter out light colors from a given palette
# def remove_light_colors(colors, max_lightness=0.80):
#     """Filter out colors that are too light based on their lightness in HLS color space.

#     Args:
#         colors: List of color hex codes or names
#         max_lightness: Maximum lightness threshold (0 to 1) for filtering colors

#     Returns:
#         List of colors that are below the lightness threshold
#     """
#     filtered = []

#     for c in colors:
#         r, g, b = mcolors.to_rgb(c)
#         h, lightness, s = colorsys.rgb_to_hls(r, g, b)

#         # Filter out colors that are too light
#         if lightness < max_lightness:
#             filtered.append(c)

#     return filtered


# glasbey = cc.glasbey
# palette_high_num_colors = remove_light_colors(glasbey)
