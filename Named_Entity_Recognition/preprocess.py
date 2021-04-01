# -*- encoding: utf-8 -*-
'''
@File   :   preprocess.py
@Time   :   2021/03/31 13:48:52
@Author :   ShareJing
@Email  :   yymmjing@gmail.com
@Desc   :   预处理数据集
'''

import json
import pandas as pd


def load_cluener2020_data(path):
    """适应simpletransformer的加载方式"""
    data = []
    labels_list = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = json.loads(line.strip())
            text = line["text"]
            label_entities = line.get("label", None)
            words = list(text)
            labels = ['O'] * len(words)
            if label_entities:
                for key, value in label_entities.items():
                    for sub_name, sub_index in value.items():
                        for start_index, end_index in sub_index:
                            assert "".join(words[start_index:end_index+1]) == sub_name
                            if start_index == end_index:
                                labels[start_index] = "S-" + key
                            else:
                                labels[start_index] = "B-" + key
                                labels[start_index+1:end_index+1] = ["I-"+key] * (len(sub_name) - 1)
            for word, label in zip(words, labels):
                data.append([idx, word, label])
                if label not in labels_list:
                    labels_list.append(label)
    data_df = pd.DataFrame(data, columns=["sentence_id", "words", "labels"])
    return data_df, labels_list


def load_cluener2020_data_1(path, mode):
    """
    加载中文cluener2020命名实体识别数据集, cluener2020原始数据集的加载方式
    https://github.com/CLUEbenchmark/CLUENER2020/blob/503a023f40550b140475110b95365980f8c0c80c/bilstm_crf_pytorch/data_processor.py#L25
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            json_d = {}
            line = json.loads(line.strip())
            text = line["text"]
            label_entities = line.get("label", None)
            words = list(text)
            labels = ['O'] * len(words)
            if label_entities:
                for key, value in label_entities.items():
                    for sub_name, sub_index in value.items():
                        for start_index, end_index in sub_index:
                            assert "".join(words[start_index:end_index+1]) == sub_name
                            if start_index == end_index:
                                labels[start_index] = "S-" + key
                            else:
                                labels[start_index] = "B-" + key
                                labels[start_index+1:end_index+1] = ["I-"+key] * (len(sub_name) - 1)
            json_d["id"] = f"{mode}_{idx}"
            json_d["content"] = " ".join(words)
            json_d["tag"] = " ".join(labels)
            json_d["raw_context"] = "".join(words)
            examples.append(json_d)
    return examples


# if __name__ == "__main__":
#     # examples = load_cluener2020_data_1("data/CLUENER2020/train.json", "train")
#     # print(len(examples))
#     # print(examples[71])
    
#     data, labels_list = load_cluener2020_data("data/CLUENER2020/train.json")
#     print(len(data))
#     print(data[:10])
#     print(labels_list)

