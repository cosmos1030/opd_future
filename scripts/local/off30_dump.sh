#!/bin/bash
set -euo pipefail
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate distil
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONPATH=/home/doyoonkim/projects/distIL/SDPO:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0
export WANDB_ENTITY=dyk6208-gwangju-institute-of-science-and-technology
R=/home/doyoonkim/projects/distIL
CKPT="$R/outputs/science/datasets/sciknoweval/chemistry/DistIL-science-chemistry--home-doyoonkim-projects-distIL-models-Qwen3-4B-Instruct-2507/global_step_30/actor"
OUT="$R/models/off30-merged"
if [ ! -f "$OUT/config.json" ]; then
  echo "=== merging off@30 (59175) ==="
  python -m verl.model_merger merge --backend fsdp --local_dir "$CKPT" --target_dir "$OUT"
fi
echo "=== val_only with per-question dump ==="
bash $R/SDPO/training/verl_training.sh \
  OFF30-dump distil datasets/sciknoweval/chemistry \
  actor_rollout_ref.model.path="$OUT" \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  trainer.val_before_train=True trainer.val_only=True \
  trainer.n_gpus_per_node=2 trainer.logger=[console] \
  trainer.total_training_steps=1 trainer.save_freq=0 \
  data.return_raw_chat=True \
  trainer.validation_data_dir=$R/valdump/off30
