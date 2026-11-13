#!/bin/bash
#SBATCH --job-name=mergeany
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/home/doyoonkim/projects/GammaOPD/logs/%x_%j.log
exec 2>&1
set -u
cd /home/doyoonkim/projects/GammaOPD
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gopd
unset ROCR_VISIBLE_DEVICES
export PYTHONPATH=/home/doyoonkim/projects/GammaOPD/verl:${PYTHONPATH:-}
mkdir -p "$(dirname "${TARGET:?}")"
[ -f "$TARGET/config.json" ] || python -m verl.model_merger merge --backend fsdp \
    --local_dir "${CKPT_SRC:?}" --target_dir "$TARGET"
ls -la "$TARGET" | head -4
echo "##### MERGED -> $TARGET #####"
