export REPO_HOME=$(pwd)  

wandb login 2f39b29deef86a6909949ed808c7406a144a0d2b 

final_folder="$REPO_HOME/verl/outputs_sci_lts"  

bash_path=$REPO_HOME/examples/grpo_trainer/sci_lts.sh


project_name=$(basename "$bash_path" .sh)
timestamp=$(date +"%Y%m%d_%H%M")
date_stamp=$(date +"%Y%m%d")
res_folder="${final_folder}/${date_stamp}/${project_name}/${timestamp}/rank_${RANK}"
echo "saving in ${res_folder}"
mkdir -p "$res_folder"

server_logging_folder="${res_folder}/server"
mkdir -p "$server_logging_folder"

output_folder="${res_folder}/outputs"
export LOG_DIR="$output_folder"

ckpt_folder="${res_folder}/ckpt"
export CKPT_DIR="$ckpt_folder"

tensorboard_folder="${res_folder}/tensorboard"
export TENSORBOARD_DIR="$tensorboard_folder"
mkdir -p "$tensorboard_folder"
echo "tb saving in ${TENSORBOARD_DIR}"

export LOG_DIR="/cpfs04/user/liutianshuo/Embodied-Planner-R1/verl/outputs_sci_normal/20250811/sci_normal/20250811_1623/rank_"
export CKPT_DIR="/cpfs04/user/liutianshuo/Embodied-Planner-R1/verl/outputs_sci_normal/20250811/sci_normal/20250811_1623/rank_/ckpt"

# source /opt/conda/etc/profile.d/conda.sh
PORT=8000
if ss -tuln | grep -q ":$PORT "; then
    echo "端口 $PORT 已被占用"
else
    echo "$PORT 未被占用"
    # conda activate scienceworld
    cd $REPO_HOME/verl/scienceworld_server
    server_cmd="python start_server.py --num_servers 8"

    nohup $server_cmd > "${server_logging_folder}/run_stdout.log" 2> "${server_logging_folder}/run_stderr.log" &
    server_pid=$!
    echo "server Process ID: $server_pid Check logs in ${server_logging_folder}/"
    # conda deactivate
fi

cd $REPO_HOME
# conda activate Embodied-Planner-R1
cmd="bash ${bash_path}"
echo "Running $cmd"

$cmd 2>&1 | tee "${res_folder}/run_stdout.log" "${res_folder}/run_stderr.log"
echo "Check logs in ${res_folder}/" 