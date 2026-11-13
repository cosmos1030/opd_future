#!/bin/bash
#SBATCH --job-name=opd_baseline_bsmatch_resume
#SBATCH --output=logs/opd_baseline_bsmatch_resume_%j.out
#SBATCH --error=logs/opd_baseline_bsmatch_resume_%j.out
#SBATCH --account=test
#SBATCH --partition=H200
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -x

timeout 20 ray stop --force || echo "ray stop timed out/skipped (known hang on this cluster, session already clean)"
export VLLM_USE_FLASHINFER_SAMPLER=0
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export RAY_memory_usage_threshold=0.99
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

EXPERIMENT_NAME="token_reward_direct_DAPO-Math-17k_Qwen3-1.7B-SFT_Qwen3-4B_7168-T_1.0-Tch_1.0-n_4-mbs_128-topk_16-topk_strategy_only_stu-rw_student_p-2026-08-23_01-08-36"
CKPT_PATH="checkpoint/${EXPERIMENT_NAME}"
RESUME_FROM_PATH="/home/doyoonkim/projects/OPD_thunlp/checkpoint/${EXPERIMENT_NAME}/global_step_80"

ray start --head
sleep 5

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=token_reward_direct \
    algorithm.grpo_outcome_weight=1.0 \
    data.shuffle=False \
    data.train_files=datasets/dapo-math-17k.parquet \
    'data.val_files=[datasets/test_data/AIME25/test.parquet, datasets/test_data/AMC23/test.parquet, datasets/test_data/AIME24/test.parquet]' \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=7168 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=model/Qwen3-1.7B-SFT \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    +actor_rollout_ref.rollout.log_prob_top_k=16 \
    +actor_rollout_ref.rollout.top_k_strategy=only_stu \
    +actor_rollout_ref.rollout.reward_weight_mode=student_p \
    +actor_rollout_ref.rollout.teacher_temperature=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=7168 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    +reward_model.reward_kwargs.enable_format_reward=False \
    reward_model.model.path=model/Qwen3-4B \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype=bfloat16 \
    reward_model.micro_batch_size_per_gpu=24 \
    custom_reward_function.path="verl/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$RESUME_FROM_PATH" \
    trainer.log_val_generations=2 \
    trainer.logger=['console'] \
    trainer.project_name=OnPolicyDistillation \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir=validation_log/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=True \

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
