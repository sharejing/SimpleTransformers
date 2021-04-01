# -*- encoding: utf-8 -*-
'''
@File   :   preprocess.py
@Time   :   2021/03/19 16:05:54
@Author :   ShareJing
@Email  :   yymmjing@gmail.com
@Desc   :   预处理数据集
'''

import pandas as pd


def load_chinese_spam_message_data(path):
    """加载中文垃圾信息数据"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            label, text = line.strip().split("\t")
            data.append([text.strip(), int(label)])
    data_df = pd.DataFrame(data, columns=["text", "labels"])
    return data_df


def load_yelp_review_data(path):
    """加载Yelp Review数据"""
    data_df = pd.read_csv(path, header=None)
    data_df[0] = (data_df[0] == 2).astype(int)
    data_df = pd.DataFrame({"text": data_df[1].replace(r"\n", " ", regex=True), "labels": data_df[0]})
    return data_df
