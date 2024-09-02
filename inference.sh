export TOKENIZERS_PARALLELISM=true 
export CUDA_VISIBLE_DEVICES=3 

for seed in 42 12 21
do
    python inference.py \
    --checkpoint declare-lab/tango2 \
    --num_steps 200 \
    --guidance 3 \
    --batch_size 48 \
    --test_file /raid2/DATA/ckpt/audiocaps_test/test_subset.jsonl \
    --test_references /raid2/DATA/ckpt/audiocaps_test/test_sub \
    --logdir ./log\
    --adjust_mode sigmoid \
    --seed $seed\
    --s1 0.6 \
    --s2 0.2 \
    --b1 1.6 \
    --b2 1.2 \

done
