#!/bin/bash
#SBATCH --job-name=hx-d
#SBATCH --partition=H200,PRO6000
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=250000
#SBATCH --time=02:00:00
#SBATCH --output=/home/doyoonkim/projects/distIL/logs/off30_dump_%j.log
#SBATCH --error=/home/doyoonkim/projects/distIL/logs/off30_dump_%j.err
srun bash /home/doyoonkim/projects/distIL/scripts/local/off30_dump.sh
