#!/bin/bash


DATA_DIR="./data"
DATA_NAME="aime24"
# DATA_NAME="MATH-500"
# DATA_NAME="gsm8k"
# DATA_NAME="minerva_math"
# DATA_NAME="olympiad_bench"

# DATA_DIR="/cpfs04/user/liutianshuo/math/deepscaler/deepscaler/data/train"
# # DATA_NAME="deepscaler"

# MODEL_PATH="/oss/public/user/liuts/model"
# MODEL_NAME="Qwen2.5_Math_1.5B"
# MODEL_NAME="Qwen2.5_Math_7B"

MODEL_PATH="/cpfs04/user/liutianshuo/math/simpleRL-reason/train/checkpoints"
# MODEL_PATH="/cpfs04/user/liutianshuo/verl/checkpoints/verl_grpo_example_math"
# /cpfs04/user/liutianshuo/verl/checkpoints/verl_grpo_example_math/Llama3B_instruct_ppo_from_checkpoint_with_system_0819/global_step_120/actor/huggingface
# MODEL_NAME="Qwen2/.5_Math_7B_sft_from_base_deepscaler_max——0519"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_max——0517"
# MODEL_NAME="Qwen2.5_Math_1.5B_sft_from_base_deepscaler_max——0517"
# MODEL_NAME="Qwen2.5_Math_1.5B_sft_from_base_deepscaler_shortest——0516"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_max_offpolicy——0521"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_shortest——0516"

# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off——0522"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_on——0522"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_norm——0522"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_lr——0522"
# MODEL_NAME="Qwen2.5_Math_1.5B_sft_from_base_deepscaler_max——0522"

# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off2——0522"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off3——0522"

# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off4——0523"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off5——0523"


# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off6——0523"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off6——0523"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off7——0523/_actor/4/"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off7——0523"

# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off8——0524/_actor/4/"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off9——0524/_actor/4/"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off8——0524"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off9——0524"


# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off12——0525/_actor/2"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off12——0525/_actor/3"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off12——0525/_actor/4"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off12——0525"

# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off16——0526/_actor/2"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off16——0526/_actor/3"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off16——0526/_actor/4"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off16——0526"

# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off17——0526/_actor/1"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off17——0526/_actor/2"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off17——0526/_actor/3"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off17——0526/_actor/4"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off17——0526"


# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off19——0526/_actor/1"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off19——0526/_actor/2"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off19——0526/_actor/3"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off19——0526/_actor/4"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off19——0526"

# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off18——0526/_actor/1"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off18——0526/_actor/2"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off18——0526/_actor/3"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off18——0526/_actor/4"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off18——0526"


# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off20——0527/_actor/1"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off20——0527/_actor/2"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off20——0527/_actor/3"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off20——0527/_actor/4"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off20——0527"

# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off22_entropy——0527/_actor/1"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off22_entropy——0527/_actor/2"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off22_entropy——0527/_actor/3"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off22_entropy——0527/_actor/4"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off22_entropy——0527"



# MODEL_NAME="Qwen2.5_Math_1.5B_sft_from_base_deepscaler——0527"

# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_1——0528/_actor/1"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_1——0528/_actor/2"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_1——0528/_actor/3"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_1——0528/_actor/4"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_1——0528/_actor/5"


# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_2——0528/_actor/1"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_2——0528/_actor/2"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_2——0528/_actor/3"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_2——0528/_actor/4"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_2——0528/_actor/5"


# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off23——0529/_actor/1"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off23——0529/_actor/2"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off23——0529/_actor/3"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off23——0529/_actor/4"
# MODEL_NAME="Qwen2.5_Math_1.5B_ppo_from_base_deepscaler_off23——0529"

# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_max——0519"
# MODEL_NAME="Qwen2.5_Math_7B_sft_from_base_deepscaler——0529"


# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_3——0530/_actor/1"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_3——0530/_actor/2"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_3——0530/_actor/3_15"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_3——0530/_actor/3_17"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_3——0530/_actor/3"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_3——0530/_actor/4_22"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_3——0530/_actor/4_24"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_3——0530/_actor/4"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_3——0530/_actor/5_29"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_3——0530/_actor/5_31"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_3——0530/_actor/5"



# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_4——0530/_actor/1"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_4——0530/_actor/2_15"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_4——0530/_actor/3"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_4——0530/_actor/3_17"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_4——0530/_actor/3_22"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_4——0530/_actor/4"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_4——0530/_actor/4_29"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_4——0530/_actor/4_31"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_4——0530/_actor/5"

# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_5——0531/_actor/4"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_5——0531/_actor/5_29"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_5——0531/_actor/5_32"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_5——0531/_actor/5_33"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_5——0531/_actor/5_34"
# MODEL_NAME="Qwen2.5_Math_7B_ppo_from_base_deepscaler_5——0531/_actor/5"

# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_4_0607/_actor/1"
# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_4_0607/_actor/2"
# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_4_0607/_actor/3"
# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_4_0607/_actor/4"
# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_4_0607/_actor/5"


# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_3_0606/_actor/1"
# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_3_0606/_actor/2"
# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_3_0606/_actor/3"
# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_3_0606/_actor/4"
# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_3_0606/_actor/5"




# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_7_0607/_actor/1"
# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_7_0607/_actor/2"
# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_7_0607/_actor/3"
# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_7_0607/_actor/4"
# MODEL_NAME="Llama-3.2-3B-Instruct_ppo_from_base_deepscaler_7_0607/_actor/5"

# MODEL_NAME="Llama-3.2-3B-Instruct_sft_from_base_deepscaler_0605/_actor/1"
# MODEL_NAME="Llama-3.2-3B-Instruct_sft_from_base_deepscaler_0605/_actor/2"
# MODEL_NAME="Llama-3.2-3B-Instruct_sft_from_base_deepscaler_0605/_actor/3"
# MODEL_NAME="Llama-3.2-3B-Instruct_sft_from_base_deepscaler_0605/_actor/4"

# MODEL_NAME="Qwen2.5_Math_7B_sft_from_checkpoint_with_system_0817/global_step_100/actor/huggingface"

MODEL_NAME="Qwen2.5-1.5B-instruct-math_math_test_0818_sft/_actor/2"
# /cpfs04/user/liutianshuo/math/simpleRL-reason/train/checkpoints/Qwen2.5-1.5B-instruct-math_math_test_0818_sft/_actor/3
# /cpfs04/user/liutianshuo/verl/checkpoints/verl_grpo_example_math/Qwen1.5b_instruct_ppo_from_checkpoint_with_system_0820/global_step_60/actor
# Qwen2.5_Math_7B_sft_from_checkpoint_with_system_0816/global_step_60/actor/huggingface
# /cpfs04/user/liutianshuo/math/simpleRL-reason/train/checkpoints/Qwen2.5-1.5B-instruct_math_test_0818_ppo/_actor/5
# ID="lhy_test_math500_no_temp"
ID="lhy_test_vllm0.6.3"

MODEL_NAME_OR_PATH=${MODEL_PATH}/${MODEL_NAME}
EVAL_DIR="./eval_results"
OUTPUT_BASE=${EVAL_DIR}/${MODEL_NAME}/${ID}
TENSOR_PARALLEL_SIZE=4
END=-1
SPLIT="test"
DATA_DIR=$DATA_DIR
DATA_NAME=$DATA_NAME

TEMPERATURE=0


# for TOKENS in $(seq 6000 500 12000); do
# for TOKENS in $(seq 4000 500 5000); do
#   OUTPUT_DIR="${OUTPUT_BASE}/length_${TOKENS}"
#   echo "Running evaluation with MAX_TOKENS_PER_CALL=$TOKENS..."
#   export CUDA_VISIBLE_DEVICES=4,5,6,7
#   python collect_and_eval.py \
#       --data_dir "$DATA_DIR" \
#       --data_name "$DATA_NAME" \
#       --model_name_or_path "$MODEL_NAME_OR_PATH" \
#       --output_dir "$OUTPUT_DIR" \
#       --end $END \
#       --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
#       --max_tokens_per_call $TOKENS \
#       --split $SPLIT \
#       --temperature $TEMPERATURE
# done

for RUN_IDX in 1; do
  TOKENS=4096
  OUTPUT_DIR="${OUTPUT_BASE}/length_${TOKENS}_run${RUN_IDX}"
  echo "Running evaluation #${RUN_IDX} with MAX_TOKENS_PER_CALL=$TOKENS..."
  export CUDA_VISIBLE_DEVICES=4,5,6,7
  python collect_and_eval.py \
      --data_dir "$DATA_DIR" \
      --data_name "$DATA_NAME" \
      --model_name_or_path "$MODEL_NAME_OR_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --end $END \
      --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
      --max_tokens_per_call $TOKENS \
      --split $SPLIT \
      --temperature $TEMPERATURE
done