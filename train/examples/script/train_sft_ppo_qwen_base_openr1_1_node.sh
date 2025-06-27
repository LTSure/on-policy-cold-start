

HOME="/cpfs04/user/liutianshuo/math/simpleRL-reason/train"


# MODEL_NAME="Qwen2.5_Math_1.5B"
# MODEL_NAME="Qwen2.5_Math_7B"
MODEL_NAME="Llama-3.2-3B-Instruct"
MODEL="/oss/public/user/liuts/model/${MODEL_NAME}"


export PYTHONPATH=$PYTHONPATH:/cpfs04/user/liutianshuo/math/simpleRL-reason/train


# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler/ppo/2/deepscaler/merge_new.json"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge_filtered_no_templete_shortest.json"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge_filtered_no_templete_max.json"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge-0-10000_filtered_no_templete_shortest.json"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0515/ppo/2/deepscaler/merge_filtered_no_templete_closed_3500_less_7000.json"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0602/ppo/2/deepscaler/merge_filtered.json"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/dafnybench_test3_filter_4.5k.json"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld-gpt4-45k_3.json"

# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld-gpt4-45k_2.json"
PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld_sft_mt_2.json"

wandb login 2f39b29deef86a6909949ed808c7406a144a0d2b 

RUN_NAME="${MODEL_NAME}_ppo_sft_alfworld_test——0620"

deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
    --pretrain $MODEL \
    --save_path $HOME/checkpoints/$RUN_NAME \
    --micro_train_batch_size 16 \
    --train_batch_size 512 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 2048 \
    --eps_clip 0.5 \
    --temperature 0.6 \
    --buffer_limit 0 \
    --n_samples_per_prompt 1 \
    --max_samples 200000 \
    --max_epochs 3 \
    --num_episodes 5 \
    --prompt_max_len 4000 \
    --generate_max_len 3000 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 1e-5 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.00 \
    --entropy_coef 0.05 \
    --prompt_data  $PROMPT_DATA \
    --input_key conversations \
    --normalize_reward \
    --flash_attn \
    --adam_offload \
    --gradient_checkpointing \
    --save_steps 200 \
    --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
    --wandb_run_name $RUN_NAME \
    --ckpt_path $HOME/checkpoints/$RUN_NAME  \
    --max_ckpt_num 20000 \
    --use_muti_turn



# # llama3.2-3B,dafnybench test 
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 64 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 256 \
#     --eps_clip 0.5 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 3 \
#     --num_episodes 5 \
#     --prompt_max_len 4500 \
#     --generate_max_len 4500 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 1e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --entropy_coef 0.05 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \


# # qwen 1.5B off24 (同off16)
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.5 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 5 \
#     --num_episodes 4 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 2e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \



# # llama3.2-3B 7, 8 （同3， 使用reward entropy）
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 128 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.5 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 3 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 0 \
#     --bf16 \
#     --actor_learning_rate 1e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --entropy_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \


# # llama3.2-3B-4 （同3， loss = actor_loss + aux_loss * self.args.aux_loss_coef - 0.05 * entropy_loss）
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 8 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 8 \
#     --rollout_batch_size 4096 \
#     --eps_clip 0.5 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 3 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 1e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --entropy_coef 0.05 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \



# # # llama3.2-3B-2
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.5 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 5 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 2e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \


# # # llama3.2-3B-1
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.5 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 5 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 2e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \
   
# # 7B 4
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 512 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 4096 \
#     --eps_clip 0.4 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 3 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 9e-6 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \


# # 7B 1
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.5 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 5 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 2e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \


# # off23（log entropy,log sft loss，same as off16）
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.5 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 5 \
#     --num_episodes 4 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 2e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \



# RUN_NAME="${MODEL_NAME}_ppo_from_base_deepscaler_off22_entropy——0527"
# # off21/off22（log entropy，same as off16）
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.5 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 5 \
#     --num_episodes 4 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 2e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \


# # off18(kl)
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.45 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 5 \
#     --num_episodes 4 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 2e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \



# # off14
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.5 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 5 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 2e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \
#     --use_decay






# # off12
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.35 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 5 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 2e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \
#     --use_decay


# # off10
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.35 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 4 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 2e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \



# # off8 
# RUN_NAME="${MODEL_NAME}_ppo_from_base_deepscaler_off8——0524"
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.3 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 4 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 8e-6 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \



# # off6 
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.30 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 4 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 9e-6 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \
#     --use_decay


# # off5
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.25 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 4 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 9e-6 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \
#     --use_decay


# # off2 
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.30 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 4 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 7e-6 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \


# # off
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 2 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.25\
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 3 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 5e-6 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \




# RUN_NAME="${MODEL_NAME}_ppo_from_base_deepscaler_lr——0522"

# # lr
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 8192 \
#     --temperature 0.6 \
#     --eps_clip 0.2\
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 1 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 7e-6 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \




















# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 8 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 16 \
#     --rollout_batch_size 32000 \
#     --temperature 0.6 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 1 \
#     --num_episodes 5 \
#     --prompt_max_len 1024 \
#     --generate_max_len 4000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 5e-6 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key input \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \
 
