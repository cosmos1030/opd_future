#!/bin/bash
#SBATCH --job-name=hx-m7
#SBATCH --output=logs/full/%x_%j.log
#SBATCH --error=logs/full/%x_%j.err
#SBATCH --account=users
#SBATCH --partition=H200
#SBATCH --gres=gpu:H200:2
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=09:00:00
cd /home/doyoonkim/projects/OPD_thunlp
bash run_mix.sh
