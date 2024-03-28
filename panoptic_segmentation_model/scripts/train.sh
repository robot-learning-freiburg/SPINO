#!/bin/bash

python_env="spino"

conda activate $python_env;
alias python="BIN_PATH/python";
export WANDB_API_KEY="WANDB_API_KEY";

wandb login --relogin WANDB_API_KEY;

CUDA_VISIBLE_DEVICES="0, 1" \
OMP_NUM_THREADS=4 \
torchrun --nproc_per_node=2 --master_addr='IP' \
                            --master_port=22001 train.py \
                            --mode=train \
                            --run_name=RUN_NAME  \
                            --project_root_dir="ROOT_DIR" \
                            --filename_defaults_config=default_config.py \
                            --filename_config=train_cityscapes_resnet.yaml \
                            --comment="Train SPINO"
#                            --resume="path-to-ckpt/epoch_X.pth" \

conda deactivate;
