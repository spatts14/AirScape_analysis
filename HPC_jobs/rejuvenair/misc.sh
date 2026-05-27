#!/bin/bash
#PBS -l walltime=00:20:00
#PBS -lselect=1:ncpus=1:mem=128gb
#PBS -N subset_celltype
#PBS -j oe

# Load production tools
module load tools/prod

# Load python and bundle
module load Biopython/1.84-foss-2024a

# Change to directory
cd /rds/general/user/sep22/home/Projects/AirScape

# Activate virtual environment
source airscape_venv/bin/activate

# Run with error logging
echo "Starting at $(date)"

# echo "Starting meta data addition..."
# python src/projects/rejuvenair/add_meta.py

# echo "Starting cleaning data addition..."
# python src/projects/rejuvenair/clean_adata.py

echo "Starting plotting addition..."
#python src/projects/rejuvenair/composition.py
#python src/projects/rejuvenair/umaps.py
python src/projects/rejuvenair/plots.py

echo "Completed at $(date)"
