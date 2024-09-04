export TOKENIZERS_PARALLELISM=true 
export CUDA_VISIBLE_DEVICES=6 
python inference.py \
    --checkpoint declare-lab/tango2 \
    --num_steps 200 \
    --guidance 3 \
    --batch_size 48 \
    --test_file /raid2/DATA/ckpt/audiocaps_test/test_subset.jsonl \
    --test_references /raid2/DATA/ckpt/audiocaps_test/test_sub \
    --logdir ./s-high\
    --adjust_mode constant \
    --seed 42\
    --s1 2.0 \
    --s2 2.0 \
    --b1 1.0 \
    --b2 1.0 \

