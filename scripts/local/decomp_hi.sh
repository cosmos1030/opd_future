#!/bin/bash
set -e
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate distil
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
cd /home/doyoonkim/projects/distIL
export VLLM_USE_FLASHINFER_SAMPLER=0
python -I probe/distil_return_decomp.py \
  --student /home/doyoonkim/projects/distIL/models/on50-merged \
  --teacher /home/doyoonkim/projects/distIL/models/Qwen3-4B-Instruct-2507 \
  --data    /home/doyoonkim/projects/distIL/SDPO/datasets/sciknoweval/chemistry/train.json \
  --n_prompts 32 --n_actions 4 --k 128 --max_new_tokens 1024 --topk 100 \
  --out probe/distil_decomp_hi.pt
