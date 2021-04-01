# # 处理中文垃圾信息分类
# CUDA_VISIBLE_DEVICES=1,2,3 python train.py \
#     --trainset_path data/Chinese_Spam_Message/train.txt \
#     --testset_path data/Chinese_Spam_Message/test.txt \
#     --model_type bert \
#     --model_name bert-base-chinese \
#     --wandb_project Chinese_Spam_Message_Classification \
#     --n_gpu 3 \
#     --num_train_epochs 3

# # 试试一些社区模型
# CUDA_VISIBLE_DEVICES=1,2,3 python train.py \
#     --trainset_path data/Chinese_Spam_Message/train.txt \
#     --testset_path data/Chinese_Spam_Message/test.txt \
#     --model_type bert \
#     --model_name hfl/chinese-bert-wwm-ext \
#     --wandb_project Chinese_Spam_Message_Classification \
#     --n_gpu 3 \
#     --num_train_epochs 3

# 试试微调之后的模型
CUDA_VISIBLE_DEVICES=1,2,3 python train.py \
    --trainset_path data/Chinese_Spam_Message/train.txt \
    --testset_path data/Chinese_Spam_Message/test.txt \
    --model_type bert \
    --model_name ../Fine_Tuning/outputs/checkpoint-756-epoch-3/ \
    --wandb_project Chinese_Spam_Message_Classification \
    --n_gpu 3 \
    --num_train_epochs 3

# 处理英文Yelp Review分类
# CUDA_VISIBLE_DEVICES=1,2,3 python train.py \
#     --trainset_path data/Yelp_Review/train.csv \
#     --testset_path data/Yelp_Review/test.csv \
#     --model_type roberta \
#     --model_name roberta-base \
#     --wandb_project Yelp_Review_Classification \
#     --n_gpu 3 \
#     --num_train_epochs 1


# 一些缓存文件存放在/root/.cache/huggingface/transformers