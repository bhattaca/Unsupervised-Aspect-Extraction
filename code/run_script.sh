#!/usr/bin/env bash
# THEANO_FLAGS="device=gpu0,floatX=float32" python train.py \
# --emb ../preprocessed_data/restaurant/w2v_embedding \
#--domain restaurant \
# -o output_dir \
python3 train.py --emb ../../data/english/embeddings/gen_english.vec.npy --domain restaurant -o out --epochs 25 --aspect-size 7 --model-name _as7
python3 train.py --emb ../../data/english/embeddings/gen_english.vec.npy --domain restaurant -o out --epochs 25 --aspect-size 14 --model-name _as14
python3 train.py --emb ../../data/english/embeddings/gen_english.vec.npy --domain restaurant -o out --epochs 25 --aspect-size 28 --model-name _as28
python3 train.py --emb ../../data/english/embeddings/gen_english.vec.npy --domain restaurant -o out --epochs 25 --aspect-size 56 --model-name _as56
