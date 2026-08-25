#!/bin/bash
#PBS -l walltime=01:0:0
#PBS -lselect=1:ncpus=1:mem=256gb
#PBS -N misc_analysis
#PBS -j oe

# Load production tools
module load tools/prod

# Load python and bundle
module load Biopython/1.84-foss-2024a

# Change to directory
cd /rds/general/user/sep22/home/Projects/AirScape_analysis

# # Activate virtual environment
# source airscape_venv/bin/activate

# Activate virtual environment
source muspan/bin/activate

# Run with error logging
echo "Starting at $(date)"

# python src/view_meta.py
# python src/PC_single_cells.py
# python src/umap.py
python src/composition_level.py
# python src/composition.py

echo "Completed at $(date)"
