# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/07/29 17:29:13
@Author  :   Sharejing
@Contact :   yymmjing@gmail.com
@Desc    :   训练阅读理解模型
'''

from config import set_args
from preprocess import load_squad2_data
from simpletransformers.question_answering import QuestionAnsweringModel
import logging


if __name__ == "__main__":
    args = set_args()

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    train_data_json = load_squad2_data(args.trainset_path)
    test_data_json = load_squad2_data(args.testset_path)

    logging.info("The len of train data: %d" % len(train_data_json))
    logging.info("The len of test data: %d" % len(test_data_json))


    model = QuestionAnsweringModel(args.model_type, args.model_name, use_cuda=True, args=vars(args), mirror="tuna")
    
    # 训练模型，并在训练时评估
    model.train_model(train_data_json, eval_data=test_data_json)
