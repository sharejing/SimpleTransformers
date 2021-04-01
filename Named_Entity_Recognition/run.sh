# 处理中文命名实体识别
CUDA_VISIBLE_DEVICES=1,2,3 python train.py \
    --trainset_path data/CLUENER2020/train.json \
    --devset_path data/CLUENER2020/dev.json \
    --testset_path data/CLUENER2020/test.json \
    --model_type bert \
    --model_name bert-base-chinese \
    --wandb_project Named_Entity_Recognition \
    --n_gpu 3 \
    --num_train_epochs 3