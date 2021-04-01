# -*- encoding: utf-8 -*-
'''
@File   :   test.py
@Time   :   2021/03/19 11:30:28
@Author :   ShareJing
@Email  :   yymmjing@gmail.com
@Desc   :   测试程序
'''

from config import set_args
from simpletransformers.classification import ClassificationModel

args = set_args()


model = ClassificationModel(args.model_type, "outputs/checkpoint-198-epoch-2/", num_labels=6, args=vars(args))


predictions, raw_outputs = model.predict(["xxxxxxxxxxxxxxxxxxx中国银行陈青林，"])
print(predictions)
print(raw_outputs)