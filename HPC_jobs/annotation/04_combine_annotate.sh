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
PROJECT_DIR="/rds/general/user/sep22/home/Projects/AirScape_analysis"
RUN_DIR="/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/2026-03-17_analysis_run"

# Change to project directory
cd "$PROJECT_DIR"

# Activate virtual environment
source airscape_venv/bin/activate

# Set directory paths
export BASE_DIR="$RUN_DIR"
export ANNOTATE_DIR="$RUN_DIR/annotate"
export CELLTYPE_SUBSET_DIR="$RUN_DIR/celltype_subset"

# Run with error logging
echo "Starting at $(date)"

if ! find "$CELLTYPE_SUBSET_DIR" -type f -name "*_annotations.csv" | grep -q .; then
	echo "No annotation CSV files found under $CELLTYPE_SUBSET_DIR"
	echo "Run the mapping job first: HPC_jobs/annotation/03_map_celltype.sh"
	exit 1
fi

# Annotate adata with cell type labels from CSV files and visualize with UMAP
python src/manual_src/annotation/04_combine_annotation.py
