#!/bin/bash
set -e
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate distil
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
cd /home/doyoonkim/projects/distIL
export VLLM_USE_FLASHINFER_SAMPLER=0
S=/home/doyoonkim/projects/distIL/models/on50-merged
T=/home/doyoonkim/projects/distIL/models/Qwen3-4B-Instruct-2507
D=/home/doyoonkim/projects/distIL/SDPO/datasets/sciknoweval/chemistry/train.json

echo "########## SMOKE ##########"
python -I probe/distil_return_decomp.py --student $S --teacher $T --data $D \
  --n_prompts 2 --n_actions 2 --k 2 --max_new_tokens 64 --topk 20 \
  --out probe/smoke.pt
echo "########## SMOKE PASSED — FULL RUN ##########"
python -I probe/distil_return_decomp.py --student $S --teacher $T --data $D \
  --n_prompts 24 --n_actions 4 --k 16 --max_new_tokens 512 --topk 100 \
  --out probe/distil_decomp.pt
