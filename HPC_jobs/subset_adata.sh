
#!/bin/bash
#PBS -l walltime=24:0:0
#PBS -lselect=1:ncpus=1:mem=256gb
#PBS -N subset_celltype
#PBS -j oe

# Load production tools
module load tools/prod

# Load python and bundle
module load Biopython/1.84-foss-2024a

# Change to directory
cd /rds/general/user/sep22/home/Projects/AirScape_analysis

# Activate virtual environment
source airscape_venv/bin/activate

# Set directory paths
export BASE_DIR="/rds/general/user/sep22/projects/phenotypingsputumasthmaticsaurorawellcomea1/live/Sara_Patti/009_ST_Xenium/output/2026-03-27_analysis_run"

# Run with error logging
echo "Starting at $(date)"

# subset adata
python src/manual_src/misc_analysis/subset_adata.py

echo "Completed at $(date)"
