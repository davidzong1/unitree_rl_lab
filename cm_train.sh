#!/bin/bash
clear
mkdir -p ./custom/checkout_point
mkdir -p ./custom/logs
mkdir -p ./custom/PlayDir
echo "----------------------------------------------------"
echo "Please input number of Environment(number of robot parallel training,default is 4096): "
read train_envs_num
if [ -z "$train_envs_num" ]; then
  train_envs_num=4096
fi
echo "You input train_envs_num: $train_envs_num"
echo "----------------------------------------------------"
echo "----------------------------------------------------"
project_id=""
INIT_PY="source/custom_lab/custom_lab/Agent/__init__.py"
mapfile -t REGISTER_IDS < <(python3 - "$INIT_PY" <<'PY'
import re, sys
path = sys.argv[1]
s = open(path, 'r', encoding='utf-8').read()
ids = []
for m in re.finditer(r'^\s*gym\.register\(', s, flags=re.M):
    start = m.start()
    i = start
    depth = 0
    block = None
    while i < len(s):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                block = s[start:i+1]
                break
        i += 1
    if block is None:
        continue
    idm = re.search(r'id\s*=\s*["\']([^"\']+)["\']', block)
    if idm:
        env_id = idm.group(1)
        if not env_id.endswith("_PLAY"):
            ids.append(env_id)
for _id in ids:
    print(_id)
PY
)

if [ ${#REGISTER_IDS[@]} -eq 0 ]; then
  echo "No unannotated \033[31mTrain Task Name\033[0m entries were found in $INIT_PY"
  SELECTED_GYM_ID=""
  exit 1
else
  echo -e "$INIT_PY has found \033[31mTrain Task Name\033[0m id:"
  for i in "${!REGISTER_IDS[@]}"; do
    idx=$((i+1))
    echo "  $idx) ${REGISTER_IDS[$i]}"
  done

  # 让用户选择一个 id，只有输入有效编号才跳出循环
  while true; do
    echo -n "Please input your choice (1-${#REGISTER_IDS[@]}): "
    read -r choice
    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#REGISTER_IDS[@]}" ]; then
      SELECTED_GYM_ID="${REGISTER_IDS[$((choice-1))]}"
      project_id=$SELECTED_GYM_ID
      echo "You choice gym id: $project_id"
      break
    else
      echo "Invalid selection: '$choice'. Please enter a number between 1 and ${#REGISTER_IDS[@]}."
    fi
  done
fi
echo "----------------------------------------------------"
echo "----------------------------------------------------"
echo "Which is choice for training logger: "
logger=""
while true; do
  echo "1.wandb    2.tensorboard    3.neptune    4.disable logging"
  read -r logger_choice
  case "$logger_choice" in
    1)
      logger="--logger=wandb"
      echo "You choice logger: wandb"
      break
      ;;
    2)
      logger="--logger=tensorboard"
      echo "You choice logger: tensorboard"
      break
      ;;
    3)
      logger="--logger=neptune"
      echo "You choice logger: neptune"
      break
      ;;
    4)
      logger=""
      echo "You choice: disable logging"
      break
      ;;
    *)
      echo "Invalid choice: '$logger_choice'. Please enter 1, 2, 3 or 4."
      ;;
  esac
done
echo "----------------------------------------------------"
echo "----------------------------------------------------"
echo "Please input number of max iteration(default is 20000): "
read num_iteration
if [ -z "$num_iteration" ]; then
  num_iteration=20000
fi
echo "You input number of iteration: $num_iteration"
echo "----------------------------------------------------"
echo "----------------------------------------------------"
headless_mode="--headless"
echo "If is headless mode(default is True): "
echo "1.True    2.False"
read headless_input
if [ -z "$headless_input" ] || [ "$headless_input" -eq 1 ]; then
  headless_mode="--headless"
  echo "You has opened headless mode."
elif [ "$headless_input" -eq 2 ]; then
  headless_mode=""
  echo "You has closed headless mode."
else
  echo "Invalid choice: '$headless_input'. Using default headless mode."
  headless_mode="--headless"
fi
echo "----------------------------------------------------"
echo "----------------------------------------------------"
# Detect Checkout point files
CHECKOUT_COMMAND=""
echo "If checkout from a checkpoint file(default is No): "
echo "1.Yes    2.No"
read checkout_select
if [ -z "$checkout_select" ]||[ "$checkout_select" -eq 2 ]; then
  echo "You has chosen not to checkout from a checkpoint file."
elif [[ ! "$checkout_select" =~ ^[0-9]+$ ]] || [ "$checkout_select" -ne 1 ]; then
  echo "Invalid choice: '$checkout_select'. Not checkout from a checkpoint file."
else
  if [ "$checkout_select" -eq 1 ]; then
    echo "Searching for checkpoint files in  custom/checkout_point..."
    CHECKOUT_DIR="./custom/checkout_point"
    mkdir -p "$CHECKOUT_DIR"
    if [ -d "$CHECKOUT_DIR" ]; then
      mapfile -t CHECKPOINT_FILES < <(find "$CHECKOUT_DIR" -type f -name "*.pt" | sort)
      if [ ${#CHECKPOINT_FILES[@]} -eq 0 ]; then
        echo "No checkpoint files found in $CHECKOUT_DIR."
      else
        echo "Found checkpoint files:"
        for i in "${!CHECKPOINT_FILES[@]}"; do
          idx=$((i+1))
          checkpoint_name="$(basename "${CHECKPOINT_FILES[$i]}")"
          echo "  $idx) ${checkpoint_name}"
        done

        while true; do
          echo -n "Please input your choice (1-${#CHECKPOINT_FILES[@]}): "
          read -r checkpoint_choice
          if [[ "$checkpoint_choice" =~ ^[0-9]+$ ]] && [ "$checkpoint_choice" -ge 1 ] && [ "$checkpoint_choice" -le "${#CHECKPOINT_FILES[@]}" ]; then
            SELECTED_CHECKPOINT="${CHECKPOINT_FILES[$((checkpoint_choice-1))]}"
            checkpoint_name="$(basename "$SELECTED_CHECKPOINT")"
            CHECKOUT_COMMAND="--resume=True --load_run=. --checkpoint=$(basename "$SELECTED_CHECKPOINT")"
            echo "You chose checkpoint file: $checkpoint_name"
            break
          else
            echo "Invalid selection: '$checkpoint_choice'. Please enter a number between 1 and ${#CHECKPOINT_FILES[@]}."
          fi
        done
      fi
    else
      echo "Checkpoint directory $CHECKOUT_DIR does not exist."
    fi
  fi
fi
echo "----------------------------------------------------"
echo "----------------------------------------------------"
GPU_COUNT=0
GPU_INFO=()
if command -v nvidia-smi >/dev/null 2>&1; then
  # 返回每行 "index, name, memory"
  mapfile -t GPU_INFO < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | sed 's/, */,/g')
  GPU_COUNT=${#GPU_INFO[@]}
else
  # 回退：使用 python 检测 torch.cuda
  GPU_COUNT=$(python3 - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)
fi
if [ "$GPU_COUNT" -gt 1 ]; then
  echo "Detected $GPU_COUNT GPUs:"
  if [ ${#GPU_INFO[@]} -gt 0 ]; then
    for entry in "${GPU_INFO[@]}"; do
      idx=$(echo "$entry" | cut -d, -f1)
      info=$(echo "$entry" | cut -d, -f2-)
      echo "  $idx) $info"
    done
  else
    for ((i=0;i<GPU_COUNT;i++)); do
      echo "  $i) cuda:$i"
    done
  fi
  # echo "  $GPU_COUNT) use All GPUs"

  echo -n "Select GPU index and press Enter to use (default use cuda0): "
  read -r gpu_choice
  if [ -z "$gpu_choice" ]; then
    gpu_choice="0"
  fi

  if [[ ! "$gpu_choice" =~ ^[0-9]+([,][0-9]+)*$ ]]; then
    echo "Invalid selection '$gpu_choice',use default GPU 0."
    gpu_choice="0"
  fi
  CUDA_VISIBLE_DEVICES_NUM=0
  CUDA_VISIBLE_DEVICES_NAME="cuda:0"
  if (( gpu_choice == GPU_COUNT )); then
    # for ((i=1;i<GPU_COUNT;i++)); do
    #     CUDA_VISIBLE_DEVICES_NAME="$CUDA_VISIBLE_DEVICES_NAME,cuda:$i"
    # done
    # echo "Using CUDA_VISIBLE_DEVICES_NAME=$CUDA_VISIBLE_DEVICES_NAME"
    echo "Detected $GPU_COUNT GPU(s). Using default device."
    CUDA_VISIBLE_DEVICES_NAME="cuda:0"
  elif ((gpu_choice > GPU_COUNT - 1)); then
    echo "Detected $GPU_COUNT GPU(s). Using default device."
    CUDA_VISIBLE_DEVICES_NAME="cuda:0"
  else
    CUDA_VISIBLE_DEVICES_NAME="cuda:$gpu_choice"
    echo "Using CUDA_VISIBLE_DEVICES_NAME=$CUDA_VISIBLE_DEVICES_NAME"
    CUDA_VISIBLE_DEVICES_NUM="$gpu_choice"
  fi
else
  echo "Detected $GPU_COUNT GPU(s). Using default device."
  CUDA_VISIBLE_DEVICES_NAME="cuda:0"
fi
echo "----------------------------------------------------"
echo "----------------------------------------------------"
echo "If Recompile Dynamic Library(default is No): "
echo "1.Yes    2.No"
COMPILE_DYNAMIC=""
read compile_dynamic_input
if [ -z "$compile_dynamic_input" ] || [ "$compile_dynamic_input" -eq 2 ]; then
  COMPILE_DYNAMIC=""
  echo "You has chosen not to recompile dynamic library."
elif [ "$compile_dynamic_input" -eq 1 ]; then
  COMPILE_DYNAMIC="--compile_dynamic=True"
  echo "You has chosen to recompile dynamic library."
else
  echo "Invalid choice: '$compile_dynamic_input'. Not recompile dynamic library."
  COMPILE_DYNAMIC=""
fi
echo "----------------------------------------------------"
echo "----------------------------------------------------"
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/source"
# export WANDB_MODE="offline"
# export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_NUM
export PHYSX_GPU_DEVICE=$CUDA_VISIBLE_DEVICES_NUM
if [[ "$gpu_choice" =~ ^[0-9]+$ ]]; then
  export CUDA_VISIBLE_DEVICES="$gpu_choice"
  CUDA_VISIBLE_DEVICES_NUM=0
  CUDA_VISIBLE_DEVICES_NAME="cuda:0"
  echo "Remapped GPU $gpu_choice to local cuda:0 via CUDA_VISIBLE_DEVICES."
fi
echo "python scripts/rsl_rl/train.py \
    --task=$project_id \
    --num_envs=$train_envs_num $logger \
    --max_iterations=$num_iteration $headless_mode \
    --device_select=$CUDA_VISIBLE_DEVICES_NAME $CHECKOUT_COMMAND \
    $COMPILE_DYNAMIC"

echo "Train will start after 3 seconds, press Ctrl+C to cancel..."
sleep 1
echo "Train will start after 2 seconds, press Ctrl+C to cancel..."
sleep 1
echo "Train will start after 1 seconds, press Ctrl+C to cancel..."
sleep 1

python scripts/rsl_rl/train.py \
    --task=$project_id \
    --num_envs=$train_envs_num $logger \
    --max_iterations=$num_iteration $headless_mode \
    --device_select=$CUDA_VISIBLE_DEVICES_NAME $CHECKOUT_COMMAND \
    $COMPILE_DYNAMIC

    