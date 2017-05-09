#!/usr/bin/env bash

python3 test.py --data_root='/Users/taras/Desktop/day2night' \
    --width=256 \
    --height=256 \
    --load_size=256 \
    --load_epoch=40 \
    --save_path='./amos_d2n_results_separate' \
    --backward_type='separate' \
    --phase='test'