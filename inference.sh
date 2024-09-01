export TOKENIZERS_PARALLELISM=true 
export CUDA_VISIBLE_DEVICES=4 

for seed in 42 1024 211
do
    python inference.py \
    --checkpoint declare-lab/tango \
    --num_steps 200 \
    --guidance 3 \
    --batch_size 48 \
    --test_file /raid2/DATA/ckpt/audiocaps_test/test_subset.jsonl \
    --test_references /raid2/DATA/ckpt/audiocaps_test/test_sub \
    --logdir ./test_mode\
    --adjust_mode sigmoid \
    --seed $seed\
    --s1 1.6 \
    --s2 1.2 \
    --b1 0.6 \
    --b2 0.2 \

done
