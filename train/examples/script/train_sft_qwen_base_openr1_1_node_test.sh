


# MODEL_NAME="Qwen2.5_Math_1.5B"
# # MODEL_NAME="Qwen2.5_Math_7B"
MODEL_NAME="Llama-3.2-3B-Instruct"
MODEL="/oss/public/user/liuts/model/${MODEL_NAME}"

# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off16_3——0526"
# MODEL="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/checkpoints/Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off16——0526/_actor/3"



export PYTHONPATH=$PYTHONPATH:/cpfs04/user/liutianshuo/math/simpleRL-reason/train
HOME="/cpfs04/user/liutianshuo/math/simpleRL-reason/train"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0602/ppo/2/deepscaler/merge_filtered.json"
# PROMPT_DATA="/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/qwen3_30b_a3b_deepscaler--0602/ppo/2/deepscaler/merge_filtered_new.json"

DATA_DIR="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/data"
# DATA_NAME="ultrafeedback_train_prefs"
# DATA_NAME="web_policy_sft"
# DATA_NAME="dafnybench_test2"
# DATA_NAME="alfworld-gpt4-45k_2"
DATA_NAME="alfworld-gpt4-45k_"
# DATA_NAME="webagent_gpt_test"


# INPUT="input"
# INPUT="chosen"
INPUT="conversations"
# INPUT="hints_removed"
# INPUT="prompt"

# OUTPUT="output"
# OUTPUT="ground_truth"
# OUTPUT="response"


PROMPT_DATA="${DATA_DIR}/${DATA_NAME}.json"

wandb login 2f39b29deef86a6909949ed808c7406a144a0d2b 


RUN_NAME="${MODEL_NAME}_sft_from_base_deepscaler_test_${DATA_NAME}_0617"

deepspeed --include localhost:0,1,2,3,4,5,6,7 --module openrlhf.cli.train_sft \
    --pretrain $MODEL \
    --save_path $HOME/checkpoints/$RUN_NAME \
    --micro_train_batch_size 2 \
    --train_batch_size 32 \
    --max_epochs 1 \
    --zero_stage 0 \
    --max_len 7000 \
    --bf16 \
    --learning_rate 5e-6 \
    --dataset  $PROMPT_DATA \
    --flash_attn \
    --adam_offload \
    --gradient_checkpointing \
    --save_steps -1 \
    --use_wandb "2f39b29deef86a6909949ed808c7406a144a0d2b" \
    --wandb_run_name $RUN_NAME \
    --ckpt_path $HOME/checkpoints/$RUN_NAME  \
    --max_ckpt_num 20000 \
    --input_key $INPUT \
    --apply_chat_template \
    --eval_split  train \
    # --output_key $OUTPUT \


 
