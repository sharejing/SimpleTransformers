# -*- encoding: utf-8 -*-
'''
@File   :   train.py
@Time   :   2021/03/18 16:32:42
@Author :   ShareJing
@Email  :   yymmjing@gmail.com
@Desc   :   训练分类模型
'''

from config import set_args
from preprocess import load_chinese_spam_message_data, load_yelp_review_data
from simpletransformers.classification import ClassificationModel
import logging
import sklearn


if __name__ == "__main__":
    args = set_args()

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    #============================================================================
    # 1. 训练中文垃圾信息分类模型
    #============================================================================
    train_df = load_chinese_spam_message_data(args.trainset_path)
    test_df = load_chinese_spam_message_data(args.testset_path)
    print(train_df.head())
    print(test_df.head())

    # 创建分类模型
    model = ClassificationModel(args.model_type, args.model_name, num_labels=6, args=vars(args))
    # model.save_model(model=model.model)  # 可以将预训练模型下载到output_dir
    
    # 训练模型，并在训练时评估
    model.train_model(train_df, eval_df=test_df, acc=sklearn.metrics.accuracy_score)


    #============================================================================
    # 2. 训练英文Yelp Review分类模型
    #============================================================================
    # train_df = load_yelp_review_data(args.trainset_path)
    # test_df = load_yelp_review_data(args.testset_path)
    # print(train_df.head())
    # print(test_df.head())

    # # model_type和model_name可以更换多种模型，具体参考：https://huggingface.co/transformers/pretrained_models.html
    # model = ClassificationModel(args.model_type, args.model_name, num_labels=2, args=vars(args))
    # model.train_model(train_df, eval_df=test_df, acc=sklearn.metrics.accuracy_score)
