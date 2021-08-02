# 处理中文垃圾信息分类
CUDA_VISIBLE_DEVICES=0 python train.py \
    --trainset_path data/SQuAD2.0/train-v2.0.json \
    --testset_path data/SQuAD2.0/dev-v2.0.json \
    --model_type bert \
    --model_name bert-base-cased \
    --wandb_project Question_Answering \
    --n_gpu 1 \
    --num_train_epochs 3
