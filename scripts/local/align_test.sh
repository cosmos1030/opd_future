#!/bin/bash
set -euo pipefail
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate distil
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONPATH=/home/doyoonkim/projects/distIL/SDPO:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0
R=/home/doyoonkim/projects/distIL
cd "$R"
B="$R/outputs/science/datasets/sciknoweval/chemistry"

merge () {  # merge <ckpt_dir> <out_dir>
  [ -f "$2/config.json" ] && { echo "already merged: $2"; return; }
  echo "=== merging $1 -> $2 ==="
  python -m verl.model_merger merge --backend fsdp --local_dir "$1/actor" --target_dir "$2"
}

ON_CKPT=$(ls -d "$B"/*on30dump*/global_step_30 2>/dev/null | head -1)
[ -z "$ON_CKPT" ] && { echo "FATAL: on@30 checkpoint not found"; exit 1; }
merge "$ON_CKPT" "$R/models/on30-merged"
merge "$B/DistIL-science-chemistry--home-doyoonkim-projects-distIL-models-Qwen3-4B-Instruct-2507/global_step_30" "$R/models/off30-merged"

python -I probe/branch_policy_align.py \
  --ref      "$R/models/Qwen3-4B-Instruct-2507" \
  --teacher  "$R/models/Qwen3-4B-Instruct-2507" \
  --arms     on="$R/models/on30-merged" off="$R/models/off30-merged" \
  --data     "$R/SDPO/datasets/sciknoweval/chemistry/train.json" \
  --n_prompts 128 --n_actions 4 --k 16 --max_new_tokens 512 --topk 100 \
  --out probe/branch_align.pt
