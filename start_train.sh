#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_DIR/venv/bin/activate"

python3 test_gpu.py
python3 download_data.py
python3 prepare_data.py

# auto-select launcher: torchrun (DDP) for multi-GPU, python3 for single GPU / CPU
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Detected $GPU_COUNT GPUs — launching with torchrun (DDP, nproc_per_node=$GPU_COUNT)"
    TRAIN_CMD="torchrun --nproc_per_node=$GPU_COUNT train.py"
else
    echo "Single GPU or CPU — launching with python3"
    TRAIN_CMD="python3 train.py"
fi

screen -dmS training bash -lc "cd $PROJECT_DIR && source venv/bin/activate && $TRAIN_CMD"
screen -dmS tensorboard bash -lc "cd $PROJECT_DIR && source venv/bin/activate && tensorboard --logdir=constellation_one_text --host=0.0.0.0 --port=6007"

screen -r training