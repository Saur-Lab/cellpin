#!/bin/bash
#SBATCH --job-name=atlas_ft_movie
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/atlas_ft_movie_%j.out
#SBATCH --error=logs/atlas_ft_movie_%j.err

# For the quick test run, add --test to the python call below.
# Full run: remove --test (or comment it out).
#TEST_FLAG="--test"

mkdir -p logs

eval "$(micromamba shell hook --shell bash)"
micromamba activate match_atlas

SCRIPT_DIR="/home/philipp.putze/github_repos/cellpin_atlas_emb/cellpin/scripts"

PYTHONUNBUFFERED=1 python "${SCRIPT_DIR}/run_atlas_finetune_movie.py" \
    --cell_type_col "Level_4" \
    --out_dir "${SCRIPT_DIR}/../outputs/atlas_finetune_movie"
#    ${TEST_FLAG}
