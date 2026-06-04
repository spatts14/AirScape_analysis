#!/bin/bash
#PBS -l walltime=08:00:00
#PBS -lselect=1:ncpus=8:mem=256gb

# Load production tools
module load tools/prod

# Load python and bundle
module load Biopython/1.84-foss-2024a

# Change to directory
cd /rds/general/user/sep22/home/Projects/AirScape_analysis

# Activate virtual environment
source airscape_venv/bin/activate

# Run
echo "Starting at annotation... $(date)"
python -m recode_st config_files/airscape.toml
