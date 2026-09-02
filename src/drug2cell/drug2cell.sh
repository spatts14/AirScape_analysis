#!/bin/bash
#PBS -l walltime=12:0:0
#PBS -l select=1:ncpus=4:mem=256gb
#PBS -N drug2cell
#PBS -j oe

# Load production tools
module load tools/prod

# Load python and bundle
module load Biopython/1.84-foss-2024a

# Change to directory
cd /rds/general/user/sep22/home/Projects/AirScape_analysis

# Activate virtual environment
source muspan/bin/activate

# install required packages
pip install pickle5

# Run with error logging
echo "Starting at $(date)"

python src/drug2cell/drug2cell_IPF.py

echo "Completed at $(date)"
