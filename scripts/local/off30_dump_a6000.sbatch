#!/bin/bash
#SBATCH --job-name=hx-d
#SBATCH --partition=A6000
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200000
#SBATCH --time=01:30:00
#SBATCH --output=/home/doyoonkim/projects/distIL/logs/off30_dump_a6000_%j.log
bash /home/doyoonkim/projects/distIL/scripts/local/off30_dump.sh
