export TOKENIZERS_PARALLELISM=true \
export CUDA_VISIBLE_DEVICES=2 \

python inference.py \
--checkpoint declare-lab/tango \
--num_steps 200 \
--guidance 3 \
--batch_size 12 \
--test_file /raid2/DATA/ckpt/audiocaps_test/test_subset.jsonl \
--test_references /raid2/DATA/ckpt/audiocaps_test/test_sub \
--logdir ./newlog \
--adjust_mode linear \
--s1 1.4 \
--s2 1.6 \
--b1 0.2 \
--b2 0.4 \
