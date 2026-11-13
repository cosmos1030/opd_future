#!/bin/bash
#SBATCH --job-name=hx-ev
#SBATCH --output=logs/full/%x_%j.log
#SBATCH --error=logs/full/%x_%j.log
#SBATCH --account=users
#SBATCH --partition=H200
#SBATCH --gres=gpu:H200:4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=02:00:00
# MATH-500 mean@16 on the last checkpoint a run left behind. save_freq=10, so if the
# job died early this still evaluates the furthest step that made it to disk.
set -x
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate verl_opd   # the eval script does not activate anything itself
export VLLM_USE_FLASHINFER_SAMPLER=0
cd /home/doyoonkim/projects/OPD_thunlp
: "${CKPT_DIR:?set CKPT_DIR to a checkpoint directory}"
STEP=${STEP:-$(ls -d "$CKPT_DIR"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)}
[ -z "$STEP" ] && { echo "FATAL: no checkpoint under $CKPT_DIR"; exit 17; }
export RESUME_FROM_PATH="$CKPT_DIR/global_step_$STEP"
echo "evaluating $RESUME_FROM_PATH"
bash eval_qwen3_gae.sh
