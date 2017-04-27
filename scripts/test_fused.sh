#!/usr/bin/env bash

python3 test.py --data_root='/hdd/datasets/horse2zebra' \
    --width=256 \
    --hight=256 \
    --load_size=256 \
    --save_path='./results_fused' \
    --backward_type='fused' \
    --phase='test' \
    --load_epoch=99 \
    --num_test_iterations=5
    #--display_id=1