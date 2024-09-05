#!/bin/bash

# 设置环境变量
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=7

python inference.py \
    --checkpoint declare-lab/tango2 \
    --num_steps 200 \
    --guidance 3 \
    --batch_size 48 \
    --test_file /raid2/DATA/ckpt/audiocaps_test/test_subset_fix.jsonl \
    --test_references /raid2/DATA/ckpt/audiocaps_test/test_sub \
    --logdir ./new_tesst\
    --adjust_mode sigmoid \
    --seed 27\
    --s1 2.6 \
    --s2 2.4 \
    --b1 0.5 \
    --b2 0.1 \
    --m 2.5
