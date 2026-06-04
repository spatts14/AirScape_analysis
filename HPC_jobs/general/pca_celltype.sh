
#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -lselect=1:ncpus=1:mem=96gb

# Load production tools
module load tools/prod

# Load python and bundle
module load Biopython/1.84-foss-2024a

# Change to directory
cd /rds/general/user/sep22/home/Projects/AirScape_analysis

# Activate virtual environment
source airscape_venv/bin/activate

# Run with error logging
echo "Starting at $(date)"

# run pseudobulk + downstream plots
#python src/projects/general/pb_celltype.py
python src/projects/general/pb_celltype_plots.py

echo "Completed at $(date)"
