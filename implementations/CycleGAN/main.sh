#!/bin/bash

python main.py \
            --data_path /root/dataset/MUBASE \
            --save_path /root/results/CycleGAN \
            --cuda_num 1 \
            --clean_data_num 10240 \
            --train_data_num 10240 \
            --val_data_num 600 \
            --batch_size 16 \
            --total_epochs 300
