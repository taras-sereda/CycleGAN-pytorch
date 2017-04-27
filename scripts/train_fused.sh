#!/usr/bin/env bash

python3 train.py --data_root='/hdd/datasets/horse2zebra' \
    --width=256 \
    --height=256 \
    --load_size=286 \
    --save_path='./h2z_results_fused' \
    --backward_type='fused' \
    --train=True \
    --phase='train' \
    --display_id=1