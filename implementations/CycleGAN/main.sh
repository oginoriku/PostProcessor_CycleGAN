#!/bin/bash

python main.py \
            --data_path /mnt/aoni04/ogino/oki_project/Oki_Update/dataset/MUBASE \
            --save_path /mnt/aoni04/ogino/oki_project/Oki_Update/results/CycleGAN \
            --cuda_num 1 \
            --clean_data_num 10 \
            --train_data_num 10 \
            --val_data_num 10 \
            --batch_size 1 \
            --total_epochs 300
