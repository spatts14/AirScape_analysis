#!/bin/bash
#PBS -l walltime=8:0:0
#PBS -lselect=1:ncpus=16:mem=512gb

# Load production tools
module load tools/prod

# Load python and bundle
module load Biopython/1.84-foss-2024a

# Change to directory
cd /rds/general/user/sep22/home/Projects/AirScape

# Activate virtual environment
source xenium_5k_venv/bin/activate

# Run
#python /rds/general/user/sep22/home/Projects/AirScape/src/manual_src/hlca_full_filt_process.py
python /rds/general/user/sep22/home/Projects/AirScape/src/manual_src/umap_hlca.py
