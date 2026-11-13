#!/bin/bash
#SBATCH --job-name=gmerge
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/home/doyoonkim/projects/GammaOPD/logs/%x_%j.log
exec 2>&1
# In-run validation is useless here: AIME24 at n=1 on a 1.7B student reports
# acc 0.0 at every single checkpoint, 14 times out of 14. The arms have to be
# compared on something with resolution, and the earlier OPD work used
# MATH-500 (baseline 0.681, SFT base 0.666), so use that.
#
#   sbatch -J m_opd   --export=ALL,ARM=small_opd,STEP=150   merge_and_eval.sh
set -u
ARM=${ARM:?}; STEP=${STEP:?}
CK=/home/doyoonkim/projects/GammaOPD/outputs/checkpoints/${ARM}/global_step_${STEP}/actor
OUT=/home/doyoonkim/projects/GammaOPD/outputs/hf/${ARM}_step${STEP}
cd /home/doyoonkim/projects/GammaOPD
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate gopd
unset ROCR_VISIBLE_DEVICES
export HF_HOME=/home/shared/huggingface
export PYTHONPATH=/home/doyoonkim/projects/GammaOPD/verl:${PYTHONPATH:-}
mkdir -p "$(dirname "$OUT")" logs

if [ ! -f "$OUT/config.json" ]; then
  python -m verl.model_merger merge --backend fsdp \
      --local_dir "$CK" --target_dir "$OUT" || exit 1
fi
ls -la "$OUT" | head
echo "##### MERGED -> $OUT #####"
