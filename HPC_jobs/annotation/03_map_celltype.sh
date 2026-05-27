#!/bin/bash
#PBS -l walltime=24:0:0
#PBS -lselect=1:ncpus=1:mem=256gb
#PBS -N subset_celltype
#PBS -j oe

# Load production tools
module load tools/prod

# Load python and bundle
module load Biopython/1.84-foss-2024a

# Project and run directories
PROJECT_DIR="/rds/general/user/sep22/home/Projects/AirScape"
RUN_DIR="/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/2026-03-17_analysis_run"

# Change to project directory
cd "$PROJECT_DIR"

# Activate virtual environment
source airscape_venv/bin/activate

# Set directory paths
export ANNOTATE_DIR="$RUN_DIR/annotate"
export CELLTYPE_SUBSET_DIR="$RUN_DIR/celltype_subset"

# Run with error logging
echo "Starting at $(date)"

# # high resolution clustering and plotting for each major cell type
export H5AD_FILE="adata_subset_Airway_epithelial_cells.h5ad"
export SUBSET="Airway_epithelial_cells"
export RESOLUTION="0.5"
export LEVEL2_ANNOTATION_DICT='{
    "0": "Ciliated cells",
    "1": "Ciliated cells",
    "2": "Goblet cells",
    "3": "Goblet cells",
    "4": "Goblet cells",
    "5": "Unknown",
    "6": "Ciliated cells",
    "7": "Basal cells",
    "8": "Basal cells",
    "9": "Basal cells",
    "10": "Secretory epithelial cells",
    "11": "Proliferating Basal cells"
}'
export LEVEL3_ANNOTATION_DICT='{
    "0": "Ciliated cells 1",
    "1": "Ciliated cells 2",
    "2": "Goblet cells 1",
    "3": "Goblet cells 2",
    "4": "Goblet cells 3",
    "5": "Unknown - Stromal cells",
    "6": "Ciliated cells 3",
    "7": "Basal cells 1",
    "8": "Basal cells 2",
    "9": "Basal cells 3",
    "10": "Secretory epithelial cells",
    "11": "Proliferating Basal cells"
}'
python src/manual_src/annotation/03_map_celltype_levels.py

# high resolution clustering and plotting for each major cell type
export H5AD_FILE="adata_subset_Alveolar_epithelial_cells.h5ad"
export SUBSET="Alveolar_epithelial_cells"
export RESOLUTION="0.3"
export LEVEL2_ANNOTATION_DICT='{
    "0": "Unknown",
    "1": "AT2 cells",
    "2": "AT1 cells",
    "3": "AT2 cells",
    "4": "AT2 cells",
    "5": "Proliferating AT2 cells",
    "6": "AT2 cells",
    "7": "AT2 cells"
}'
export LEVEL3_ANNOTATION_DICT='{
    "0": "Intermediate AT cells and unknown",
    "1": "AT2 cells 1",
    "2": "AT1 cells",
    "3": "AT2 cells 2",
    "4": "AT2 cells 3",
    "5": "Proliferating AT2 cells",
    "6": "AT2 cells 4",
    "7": "AT2 cells 5"
}'
python src/manual_src/annotation/03_map_celltype_levels.py

# high resolution clustering and plotting for each major cell type
export H5AD_FILE="adata_subset_Endothelial_cells.h5ad"
export SUBSET="Endothelial_cells"
export RESOLUTION="0.5"
export LEVEL2_ANNOTATION_DICT='{
    "0": "Unknown",
    "1": "Blood endothelial cells - unclassified",
    "2": "Capillary endothelial cells",
    "3": "Pulmonary artery endothelial cell",
    "4": "Pulmonary vein endothelial cell",
    "5": "Pulmonary vein endothelial cell",
    "6": "Lymphatic endothelial cells",
    "7": "Aerocytes"
}'
export LEVEL3_ANNOTATION_DICT='{    "0": "Unknown",
    "1": "Blood endothelial cells - unclassified",
    "2": "Capillary endothelial cells",
    "3": "Pulmonary artery endothelial cell",
    "4": "Pulmonary vein endothelial cell 1",
    "5": "Pulmonary vein endothelial cell 2",
    "6": "Lymphatic endothelial cells",
    "7": "Aerocytes"
}'
python src/manual_src/annotation/03_map_celltype_levels.py

# high resolution clustering and plotting for each major cell type
export H5AD_FILE="adata_subset_Immune_cells.h5ad"
export SUBSET="Immune_cells"
export RESOLUTION="1.5"
export LEVEL2_ANNOTATION_DICT='{
    "0": "Mast cells",
    "1": "Monocytes",
    "2": "Unknown",
    "3": "Unknown",
    "4": "Lipid-associated macrophages",
    "5": "Dendritic cells",
    "6": "Mast cells",
    "7": "CD4+ T cells",
    "8": "Dendritic cells",
    "9": "Plasma cells",
    "10": "Macrophages",
    "11": "Macrophages",
    "12": "T cells",
    "13": "Unknown",
    "14": "Interstitial Macrophages",
    "15": "Unknown",
    "16": "Plasma cells",
    "17": "Airway/Alveolar macrophages",
    "18": "CD8+ T cells",
    "19": "CD8+ T cells",
    "20": "B cells",
    "21": "Plasma cells",
    "22": "Neutrophils",
    "23": "NK cells",
    "24": "CD4+ T cells",
    "25": "Interstitial Macrophages",
    "26": "Interstitial Macrophages",
    "27": "Airway/Alveolar macrophages",
    "28": "Unknown"
}'
export LEVEL3_ANNOTATION_DICT='{
    "0": "Mast cells 1",
    "1": "Monocytes",
    "2": "Unknown",
    "3": "Unknown - megakaryocyte",
    "4": "Lipid-associated macrophages",
    "5": "Dendritic cells 1",
    "6": "Mast cells 2",
    "7": "CD4+ T cells 1",
    "8": "Dendritic cells 2",
    "9": "Plasma cells 1",
    "10": "Macrophages - unclassified 1",
    "11": "Macrophages - unclassified 2",
    "12": "CD4+CD8+ T cells",
    "13": "Unknown - Stromal cells",
    "14": "Interstitial Macrophages 1",
    "15": "Unknown - Profilerating cells",
    "16": "Plasma cells 2",
    "17": "Airway/Alveolar macrophages 1",
    "18": "CD8+ T cells 1",
    "19": "CD8+ T cells 2",
    "20": "B cells",
    "21": "Plasma cells 3",
    "22": "Neutrophils",
    "23": "NK cells",
    "24": "CD4+ T cells 2",
    "25": "Interstitial Macrophages 2",
    "26": "Interstitial Macrophages 3",
    "27": "Airway/Alveolar macrophages 2",
    "28": "Unknown - 16S"
}'
python src/manual_src/annotation/03_map_celltype_levels.py

# high resolution clustering and plotting for each major cell type
export H5AD_FILE="adata_subset_Stromal_cells.h5ad"
export SUBSET="Stromal_cells"
export RESOLUTION="0.5"
export LEVEL2_ANNOTATION_DICT='{
    "0": "Unknown",
    "1": "SMC",
    "2": "Adventitial fibroblasts",
    "3": "Pericytes",
    "4": "CTHRC1+ fibroblasts",
    "5": "SMC",
    "6": "Adventitial fibroblasts",
    "7": "Alveolar fibroblasts",
    "8": "SMC",
    "9": "Alveolar fibroblasts",
    "10": "Unknown",
    "11": "SMC",
    "12": "Lipo-fibroblasts"
}'
export LEVEL3_ANNOTATION_DICT='{
    "0": "Unknown - APOD+",
    "1": "SMC 1",
    "2": "Adventitial fibroblasts 1",
    "3": "Pericytes",
    "4": "CTHRC1+ fibroblasts",
    "5": "SMC 2 - DES hi",
    "6": "Adventitial fibroblasts 2",
    "7": "Alveolar fibroblasts 1",
    "8": "SMC 3",
    "9": "Alveolar fibroblasts 2",
    "10": "Unknown - Basal epithelial cells",
    "11": "SMC 4 - MYH11 hi",
    "12": "Lipo-fibroblasts - PLIN1+ FABP4+"
}'
python src/manual_src/annotation/03_map_celltype_levels.py


echo "Completed at $(date)"
