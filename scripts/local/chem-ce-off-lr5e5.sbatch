#!/bin/bash
#SBATCH --job-name=hx-s
#SBATCH --partition=H200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=400000M
#SBATCH --time=06:00:00
#SBATCH --output=/home/doyoonkim/projects/distIL/logs/%x_%j.log

srun bash -c 'source /opt/anaconda3/2022.05/etc/profile.d/conda.sh; conda activate distil; unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; export WANDB_ENTITY=dyk6208-gwangju-institute-of-science-and-technology; export DISTIL_PG_MODE=off; export DISTIL_DIV=ce; export VLLM_USE_FLASHINFER_SAMPLER=0; export PYTHONPATH=/home/doyoonkim/projects/distIL/SDPO:${PYTHONPATH:-}; cd /home/doyoonkim/projects/distIL; bash SDPO/training/verl_training.sh DistIL-science-chemistry-chem-ce-off-lr5e5 distil datasets/sciknoweval/chemistry   data.train_batch_size=32 trainer.group_name=DistIL-science   trainer.n_gpus_per_node=2   actor_rollout_ref.rollout.n=8   actor_rollout_ref.model.path=/home/doyoonkim/projects/distIL/models/Qwen3-4B-Instruct-2507   actor_rollout_ref.actor.optim.lr=5e-5   actor_rollout_ref.actor.optim.lr_warmup_steps=10   actor_rollout_ref.actor.ppo_mini_batch_size=32   actor_rollout_ref.actor.self_distillation.distillation_topk=100   actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01   actor_rollout_ref.actor.self_distillation.include_environment_feedback=False   actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True   algorithm.rollout_correction.rollout_is=token   actor_rollout_ref.rollout.val_kwargs.n=16   actor_rollout_ref.rollout.val_kwargs.temperature=0.6   trainer.total_training_steps=30 trainer.save_freq=15 trainer.logger=[console]   data.return_raw_chat=True   actor_rollout_ref.rollout.gpu_memory_utilization=0.5   trainer.validation_data_dir=/home/doyoonkim/projects/distIL/valdump/chem-ce-off-lr5e5'
