# -*- encoding: utf-8 -*-
'''
@File   :   train.py
@Time   :   2021/03/31 15:01:06
@Author :   ShareJing
@Email  :   yymmjing@gmail.com
@Desc   :   训练命名实体识别模型
'''

from config import set_args
from preprocess import load_cluener2020_data
from simpletransformers.ner import NERModel
import sklearn
import logging


# 可从训练集中得出
labels_list = ["B-company", "I-company", 'O', "B-name", "I-name", 
               "B-game", "I-game", "B-organization", "I-organization",
               "B-movie", "I-movie", "B-position", "I-position",
               "B-address", "I-address", "B-government", "I-government",
               "B-scene", "I-scene", "B-book", "I-book",
               "S-company", "S-address", "S-name", "S-position"]


if __name__ == "__main__":
    args = set_args()

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    #============================================================================
    # 1. 训练中文命名实体识别模型
    #============================================================================
    train_df, _ = load_cluener2020_data(args.trainset_path)
    dev_df, _ = load_cluener2020_data(args.devset_path)
    print(train_df.head())
    print(dev_df.head())

    # 创建命名实体识别模型
    model = NERModel(args.model_type, args.model_name, labels=labels_list, args=vars(args))
    # model.save_model(model=model.model)  # 可以将预训练模型下载到output_dir
    
    # 训练模型，并在训练时评估
    model.train_model(train_df, eval_data=dev_df)