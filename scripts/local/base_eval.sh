#!/bin/bash
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate distil
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONPATH=/home/doyoonkim/projects/distIL/SDPO:${PYTHONPATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0
export WANDB_ENTITY=dyk6208-gwangju-institute-of-science-and-technology
bash /home/doyoonkim/projects/distIL/SDPO/training/verl_training.sh \
  BASE-eval-chemistry-Qwen3-4B distil datasets/sciknoweval/chemistry \
  actor_rollout_ref.model.path=/home/doyoonkim/projects/distIL/models/Qwen3-4B-Instruct-2507 \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  trainer.val_before_train=True trainer.val_only=True \
  trainer.n_gpus_per_node=2 trainer.logger=[console] \
  trainer.total_training_steps=1 trainer.save_freq=0 \
  data.return_raw_chat=True \
  trainer.validation_data_dir=/home/doyoonkim/projects/distIL/valdump/base
