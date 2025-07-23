set -x

export WANDB_API_KEY=2f39b29deef86a6909949ed808c7406a144a0d2b 

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

train_data_size=16
val_data_size=128
group_size=8
N_GPUS_PER_NODE=8

# train_data_size=1
# val_data_size=1
# group_size=2
# N_GPUS_PER_NODE=1

# MODEL_PATH="/oss/public/user/liuts/model/Llama-3.2-3B-Instruct"
# MODEL_PATH="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/checkpoints/Llama-3.2-3B-Instruct_sft_alfworld_test_0623/_actor/3"
MODEL_PATH="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/checkpoints/Llama-3.2-3B-Instruct_ppo_sft_alfworld_muti_turn_0623/_actor/3"

RAY_DEBUG=False

# RUN_NAME="grpo_qwen2.5_1.5b_pposft0710"
RUN_NAME="grpo_Llama-3.2-3B_pposft0711"

wandb login 2f39b29deef86a6909949ed808c7406a144a0d2b 

# ray stop --force
# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=8192 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=40 \
    env.rollout.n=$group_size \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@ \
    trainer.val_only=True \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=/cpfs04/user/liutianshuo/verl-agent/checkpoints/verl_agent_alfworld/grpo_llama3.2_3b_pposft_0710/global_step_150 \
