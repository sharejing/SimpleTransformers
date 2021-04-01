# -*- encoding: utf-8 -*-
'''
@File   :   train.py
@Time   :   2021/04/01 21:11:48
@Author :   ShareJing
@Email  :   yymmjing@gmail.com
@Desc   :   微调语言模型
'''

from simpletransformers.language_modeling import LanguageModelingModel
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
"reprocess_input_data": True,
"overwrite_output_dir": True,
"num_train_epochs": 3,
"n_gpu": 4
}

model = LanguageModelingModel("bert", "bert-base-chinese", args=train_args, use_cuda=True)

model.train_model("data/train.txt", eval_file="data/test.txt")

model.eval_model("data/test.txt")