#!/usr/bin/env bash

python3 train.py --data_root='/hdd/datasets/AMOS/AMOS_clean' \
    --width=128 \
    --height=128 \
    --load_size=142 \
    --save_path='./amos_d2n_results_separate' \
    --backward_type='separate' \
    --train=True \
    --phase='train' \
    --display_id=3