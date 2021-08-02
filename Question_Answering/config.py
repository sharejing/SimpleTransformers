# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2021/07/29 17:18:33
@Author  :   Sharejing
@Contact :   yymmjing@gmail.com
@Desc    :   模型训练超参数以及模型存储位置，其他参数可以参考：
             https://simpletransformers.ai/docs/usage/#task-specific-models
'''


import argparse


def data_config(parser):
    parser.add_argument("--trainset_path", type=str, default="data/SQuAD2.0/train-v2.0.json",
                        help="训练集路径")
    parser.add_argument("--testset_path", type=str, default="data/SQuAD2.0/dev-v2.0.json",
                        help="测试集路径")
    parser.add_argument("--reprocess_input_data", type=bool, default=True,
                        help="如果为True，则即使cache_dir中存在输入数据的缓存文件，也将重新处理输入数据")
    parser.add_argument("--overwrite_output_dir", type=bool, default=True,
                        help="如果为True，则训练后的模型将保存到ouput_dir，并将覆盖同一目录中的现有已保存模型")
    parser.add_argument("--use_cached_eval_features", type=bool, default=True,
                        help="训练期间的评估使用缓存特征，将此设置为False将导致在每个评估步骤中重新计算特征")
    parser.add_argument("--output_dir", type=str, default="outputs/",
                        help="存储所有输出，包括模型checkpoints和评估结果")
    parser.add_argument("--best_model_dir", type=str, default="outputs/best_model/",
                        help="保存评估过程中的最好模型")
    return parser


def model_config(parser):
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="模型支持的最大序列长度")
    parser.add_argument("--model_type", type=str, default="bert",
                        help="模型类型bert/roberta")
    parser.add_argument("--model_name", type=str, default="bert-base-cased",
                        help="选择使用哪个预训练模型")
    parser.add_argument("--manual_seed", type=int, default=610,
                        help="为了产生可重现的结果，需要设置随机种子")
    return parser


def train_config(parser):
    parser.add_argument("--evaluate_during_training", type=bool, default=True,
                        help="设置为True以在训练模型时执行评估，确保评估数据已传递到训练方法")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="模型训练迭代数")
    parser.add_argument("--evaluate_during_training_steps", type=int, default=10000,
                        help="在每个指定的step上执行评估，checkpoint和评估结果将被保存")
    parser.add_argument("--wandb_project", type=str, default="Question Answering",
                        help="W＆B项目的名称，这会将所有超参数值、训练损失和评估指标记录到给定的项目中")
    parser.add_argument("--wandb_kwargs", type=dict, default={"name": "bert"},
                        help="")
    parser.add_argument("--save_eval_checkpoints", type=bool, default=True)
    parser.add_argument("--save_model_every_epoch", type=bool, default=True,
                        help="每次epoch保存模型")
    parser.add_argument("--n_gpu", type=int, default=1,
                        help="训练时使用的GPU个数")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    return parser


def set_args():
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = train_config(parser)

    args = parser.parse_args()
    return args
