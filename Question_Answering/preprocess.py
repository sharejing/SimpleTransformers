# -*- encoding: utf-8 -*-
'''
@File    :   preprocess.py
@Time    :   2021/07/28 18:25:58
@Author  :   Sharejing
@Contact :   yymmjing@gmail.com
@Desc    :   None
'''
import json


def load_squad2_data(path):
    """
    加载squad2.0数据集以适应Simpletransformer
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    parsed_data = []

    for sample in data:
        paragraphs = sample["paragraphs"]
        for paragraph in paragraphs:
            context = paragraph["context"]
            qas = paragraph["qas"]
            new_qas = []
            for qa in qas:
                id = qa["id"]
                is_impossible = qa["is_impossible"]
                question = qa["question"]
                if not is_impossible:
                    answers = [qa["answers"][0]]
                else:
                    answers = []

                new_qas.append({
                    "id": id,
                    "is_impossible": is_impossible,
                    "question": question,
                    "answers": answers
                })
            new_sample = {
                "context": context,
                "qas": new_qas
            }
            parsed_data.append(new_sample)

    return parsed_data
