# -*- encoding: utf-8 -*-
'''
@File   :   test.py
@Time   :   2021/03/31 16:22:02
@Author :   ShareJing
@Email  :   yymmjing@gmail.com
@Desc   :   测试程序
'''

from config import set_args
from simpletransformers.ner import NERModel
from train import labels_list
import numpy as np
from scipy.special import softmax


args = set_args()

model = NERModel(args.model_type, "outputs/checkpoint-500/", labels=labels_list, args=vars(args))

sentence = " ".join("彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户")
# print(sentence)

sentences = [sentence]

predictions, raw_outputs = model.predict(sentences)
print(predictions)

# More detailed preditctions
for idx, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
    print("\n___________________________")
    print("Sentence: ", sentences[idx])
    for pred, out in zip(preds, outs):
        key = list(pred.keys())[0]
        new_out = out[key]
        preds = list(softmax(np.mean(new_out, axis=0)))
        print(key, pred[key], preds[np.argmax(preds)], preds)