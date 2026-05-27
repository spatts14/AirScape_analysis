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
python src/manual_src/annotation/02_celltype_level_resolution.py

# high resolution clustering and plotting for each major cell type
export H5AD_FILE="adata_subset_Alveolar_epithelial_cells.h5ad"
export SUBSET="Alveolar_epithelial_cells"
python src/manual_src/annotation/02_celltype_level_resolution.py

# high resolution clustering and plotting for each major cell type
export H5AD_FILE="adata_subset_Endothelial_cells.h5ad"
export SUBSET="Endothelial_cells"
python src/manual_src/annotation/02_celltype_level_resolution.py

# high resolution clustering and plotting for each major cell type
export H5AD_FILE="adata_subset_Immune_cells.h5ad"
export SUBSET="Immune_cells"
python src/manual_src/annotation/02_celltype_level_resolution.py

# high resolution clustering and plotting for each major cell type
export H5AD_FILE="adata_subset_Stromal_cells.h5ad"
export SUBSET="Stromal_cells"
python src/manual_src/annotation/02_celltype_level_resolution.py


echo "Completed at $(date)"
