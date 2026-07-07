#!/bin/bash
#SBATCH --job-name=atlas_match
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/atlas_match_%j.out
#SBATCH --error=logs/atlas_match_%j.err

mkdir -p logs

eval "$(micromamba shell hook --shell bash)"
micromamba activate match_atlas

python /home/philipp.putze/github_repos/cellpin_atlas_emb/cellpin/scripts/run_atlas_match.py
