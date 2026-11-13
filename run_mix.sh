#!/bin/bash
# OPD on teacher-mixed rollouts.
#
# Only the rollout changes: tokens are sampled from (1-a) p_S + a p_T via a vLLM
# logits processor. The loss is untouched -- its 3D ratio is over top-K candidates,
# not the sampled token, so there is no slot for MiniLLM's per-token w and none is
# applied. This arm is therefore "OPD on teacher-mixed rollouts", NOT MiniLLM's
# teacher_mixed_alpha; it cannot settle whether that recipe works.
#
# Mixing is capped to the first OPD_MIX_MAX_TOKENS response tokens so the teacher's
# KV cache stays bounded (0.1406 MiB/token/slot).
set -x
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate verl_opd

export PYTHONPATH=/home/doyoonkim/projects/opd_horizon:$PYTHONPATH
export OPD_MIX_TEACHER=/home/doyoonkim/projects/OPD_thunlp/model/Qwen3-4B
export OPD_MIX_ALPHA=${OPD_MIX_ALPHA:-0.2}
export OPD_MIX_MAX_TOKENS=${OPD_MIX_MAX_TOKENS:-512}
export OPD_MIX_PROMPT_CAP=1024

export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}

# This node is H200 NVL: NVLink bonds GPUs in groups of four, and SLURM hands out the
# lowest free indices, which straddle a group boundary as often as not. Ask for one
# spare and keep an NV-bonded pair, so FSDP all-gather rides NVLink instead of PCIe.
if [ -z "$OPD_NO_NVL_PICK" ]; then
  PICK=$(python3 - <<'EOPY'
import re, subprocess
m = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout
rows = [l.split() for l in m.splitlines() if re.match(r"^GPU\d+\s", l)]
link = {}
for r in rows:
    i = int(r[0][3:])
    for j, cell in enumerate(r[1: 1 + len(rows)]):
        if cell.startswith("NV"):
            link[(i, j)] = cell
pair = next(iter(sorted(link)), None)
print(",".join(map(str, pair)) if pair else "")
EOPY
)
  if [ -n "$PICK" ]; then
    export CUDA_VISIBLE_DEVICES=$PICK
    echo "[nvl] using NV-bonded local GPUs $PICK of ${CUDA_VISIBLE_DEVICES:-all}"
  else
    echo "[nvl] WARNING: no NV-bonded pair among the allocated GPUs; falling back"
  fi
fi
export MAX_STEPS=${MAX_STEPS:-50}
export LR=${LR:-3e-6}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}   # bounds teacher KV: 128*1536*0.1406MiB = 27 GiB
export GPU_UTIL=${GPU_UTIL:-0.40}          # leaves room for teacher weights + KV + FSDP
export ROLLOUT_SEED=${ROLLOUT_SEED:-0}
export SMOKE_STEPS=$MAX_STEPS

export EXTRA_ARGS="actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
 +actor_rollout_ref.rollout.engine_kwargs.vllm.logits_processors=['opd_mix:TeacherMixLogitsProcessor']"

bash full_h200.sh
