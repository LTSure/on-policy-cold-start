
wandb login 2f39b29deef86a6909949ed808c7406a144a0d2b 

# RUN_NAME=Qwen2.5_Math_1.5B_ppo_from_base_deepscaler——0517
# RUN_NAME=Qwen2.5_Math_7B_ppo_from_base_deepscaler_max_offpolcy——0521
# RUN_NAME=Qwen2.5-7B_ppo_from_base_openr1
HOME="/cpfs04/user/liutianshuo/math/simpleRL-reason/train"
# MODEL="/oss/public/user/liuts/model/Qwen2.5_1.5B"
# MODEL="/oss/public/user/liuts/model/Qwen2.5_Math_7B"

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
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld-gpt4-45k_2.json"
PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data/alfworld_sft_mt_2.json"

RUN_NAME="${MODEL_NAME}_ppo_sft_alfworld_muti_turn——0625"


# llama3.2-3B,alfworld muti turn 0625
deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
    --pretrain $MODEL \
    --save_path $HOME/checkpoints/$RUN_NAME \
    --micro_train_batch_size 1 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 256 \
    --eps_clip 0.5 \
    --temperature 0.6 \
    --buffer_limit 0 \
    --n_samples_per_prompt 1 \
    --max_samples 200000 \
    --max_epochs 1 \
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



# # llama3.2-3B,alfworld muti turn 0624
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 1 \
#     --train_batch_size 32 \
#     --micro_rollout_batch_size 1 \
#     --rollout_batch_size 256 \
#     --eps_clip 0.4 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 3 \
#     --num_episodes 5 \
#     --prompt_max_len 4000 \
#     --generate_max_len 3000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 1e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --entropy_coef 0.05 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key conversations \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \
#     --use_muti_turn


# # llama3.2-3B,alfworld muti turn 0623
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 1 \
#     --train_batch_size 32 \
#     --micro_rollout_batch_size 1 \
#     --rollout_batch_size 256 \
#     --eps_clip 0.5 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 3 \
#     --num_episodes 5 \
#     --prompt_max_len 4000 \
#     --generate_max_len 3000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 1e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --entropy_coef 0.05 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key conversations \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \
#     --use_muti_turn




# # llama3.2-3B,alfworld test 0614(0615)
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 16 \
#     --train_batch_size 512 \
#     --micro_rollout_batch_size 16 \
#     --rollout_batch_size 4096 \
#     --eps_clip 0.5 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 3 \
#     --num_episodes 5 \
#     --prompt_max_len 2700 \
#     --generate_max_len 1600 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 1e-5 \
#     --critic_learning_rate 9e-6 \
#     --init_kl_coef 0.00 \
#     --entropy_coef 0.05 \
#     --prompt_data  $PROMPT_DATA \
#     --input_key conversations \
#     --normalize_reward \
#     --flash_attn \
#     --adam_offload \
#     --gradient_checkpointing \
#     --save_steps 200 \
#     --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
#     --wandb_run_name $RUN_NAME \
#     --ckpt_path $HOME/checkpoints/$RUN_NAME  \
#     --max_ckpt_num 20000 \



# # llama3.2-3B,alfworld test 0613
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 512 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 4096 \
#     --eps_clip 0.5 \
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 3 \
#     --num_episodes 5 \
#     --prompt_max_len 3000 \
#     --generate_max_len 200 \
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






# RUN_NAME="${MODEL_NAME}_ppo_from_base_deepscaler_6_0607"


# # llama3.2-3B-6 （同3， loss = actor_loss + aux_loss * self.args.aux_loss_coef - 0.1 * entropy_loss）
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
#     --entropy_coef 0.1 \
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



# # llama3.2-3B-5 onpolicy
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 8 \
#     --train_batch_size 256  \
#     --micro_rollout_batch_size 8 \
#     --rollout_batch_size 4096 \
#     --eps_clip 0.5\
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 1 \
#     --num_episodes 5 \
#     --prompt_max_len 2048 \
#     --generate_max_len 7000 \
#     --zero_stage 3 \
#     --bf16 \
#     --actor_learning_rate 1e-6 \
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




# # # llama3.2-3B-3
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




# RUN_NAME="${MODEL_NAME}_ppo_from_base_deepscaler_5——0531"

# # 7B 5(同 7B 2，但是增加断点)
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
#     --actor_learning_rate 1e-5 \
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


# # 7B 3(同 7B 2，但是增加断点)
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
#     --actor_learning_rate 1e-5 \
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



# 7B 2
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
#     --actor_learning_rate 1e-5 \
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



# # off23
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.4 \
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

# # off20
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


# # off19（kl）
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
#     --use_decay


# RUN_NAME="${MODEL_NAME}_ppo_from_base_deepscaler_off16——0526"
# # off16
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



# RUN_NAME="${MODEL_NAME}_ppo_from_base_deepscaler_off17——0526"
# # off17
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.6 \
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




# # off15
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 256 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.6 \
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


# # off13 
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
#     --actor_learning_rate 1e-5 \
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


# # off11
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
#     --actor_learning_rate 3e-5 \
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





# # off9
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 512 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 2048 \
#     --eps_clip 0.32 \
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


# # off7 
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
#     --actor_learning_rate 1e-5 \
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


# # off4
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
#     --use_decay




# # off3
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 512 \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 1024 \
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





# # on
# deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft_ppo \
#     --pretrain $MODEL \
#     --save_path $HOME/checkpoints/$RUN_NAME \
#     --micro_train_batch_size 4 \
#     --train_batch_size 128  \
#     --micro_rollout_batch_size 4 \
#     --rollout_batch_size 16384 \
#     --eps_clip 0.15\
#     --temperature 0.6 \
#     --buffer_limit 0 \
#     --n_samples_per_prompt 1 \
#     --max_samples 200000 \
#     --max_epochs 1 \
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



# RUN_NAME="${MODEL_NAME}_ppo_from_base_deepscaler_norm——0522"

# # norm
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
 
