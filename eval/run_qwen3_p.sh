#!/bin/bash


echo "All processes start."

NUM=32

# CUDA_VISIBLE_DEVICES=0,1 python qwen3_collect_deepscaler.py --start 0 --end 2500 --tensor_parallel_size 2 --n $NUM &
# CUDA_VISIBLE_DEVICES=2,3 python qwen3_collect_deepscaler.py --start 2500 --end 5000 --tensor_parallel_size 2 --n $NUM &
# CUDA_VISIBLE_DEVICES=4,5 python qwen3_collect_deepscaler.py --start 5000 --end 7500 --tensor_parallel_size 2 --n $NUM &
# CUDA_VISIBLE_DEVICES=6,7 python qwen3_collect_deepscaler.py --start 7500 --end 10000 --tensor_parallel_size 2 --n $NUM &


# CUDA_VISIBLE_DEVICES=0,1 python qwen3_collect_deepscaler.py --start 10000 --end 12500 --tensor_parallel_size 2 --n $NUM &
# CUDA_VISIBLE_DEVICES=2,3 python qwen3_collect_deepscaler.py --start 12500 --end 15000 --tensor_parallel_size 2 --n $NUM &
# CUDA_VISIBLE_DEVICES=4,5 python qwen3_collect_deepscaler.py --start 15000 --end 17500 --tensor_parallel_size 2 --n $NUM &
# CUDA_VISIBLE_DEVICES=6,7 python qwen3_collect_deepscaler.py --start 17500 --end 20000 --tensor_parallel_size 2 --n $NUM &


CUDA_VISIBLE_DEVICES=0,1 python qwen3_collect_deepscaler.py --start 20000 --end 22500 --tensor_parallel_size 2 --n $NUM &
CUDA_VISIBLE_DEVICES=2,3 python qwen3_collect_deepscaler.py --start 22500 --end 25000 --tensor_parallel_size 2 --n $NUM &
CUDA_VISIBLE_DEVICES=4,5 python qwen3_collect_deepscaler.py --start 25000 --end 27500 --tensor_parallel_size 2 --n $NUM &
CUDA_VISIBLE_DEVICES=6,7 python qwen3_collect_deepscaler.py --start 27500 --end 30000 --tensor_parallel_size 2 --n $NUM &


# CUDA_VISIBLE_DEVICES=0,1 python qwen3_collect_deepscaler.py --start 30000 --end 32500 --tensor_parallel_size 2 --n $NUM &
# CUDA_VISIBLE_DEVICES=2,3 python qwen3_collect_deepscaler.py --start 32500 --end 35000 --tensor_parallel_size 2 --n $NUM &
# CUDA_VISIBLE_DEVICES=4,5 python qwen3_collect_deepscaler.py --start 35000 --end 37500 --tensor_parallel_size 2 --n $NUM &
# CUDA_VISIBLE_DEVICES=6,7 python qwen3_collect_deepscaler.py --start 37500 --end 40000 --tensor_parallel_size 2 --n $NUM &


# CUDA_VISIBLE_DEVICES=6,7 python qwen3_collect_deepscaler.py --start 40000 --end -1 --tensor_parallel_size 2 --n $NUM

wait

echo "All processes finished."



