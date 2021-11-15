# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2021/11/11 09:55:03
@Author  :   Sharejing
@Contact :   yymmjing@gmail.com
@Desc    :   使用simpletransformers来训练模型，transformers来单条测试
'''

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("./outputs/best_model")
model = AutoModelForTokenClassification.from_pretrained("./outputs/best_model")
cate2ZH = {
    "person": "人物",
    "article": "文章",
    "industry": "行业",
    "organization": "机构",
    "product": "产品",
    "brand": "品牌",
    "research_report": "研报",
    "business": "业务"
}

print("模型启动成功.....")


def predict_entity(query):
    """
    给定query，预测其中的entity
    """
    inputs = tokenizer(query, return_tensors="pt")
    tokens = inputs.tokens()
    positions = list(range(len(tokens)))
    outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    entity = ""
    cate = ""
    postion = [-1, -1]
    results = []  # 存储实体及类型
    for token, pos, prediction in zip(tokens, positions, predictions[0].numpy()):
        pos -= 1  # 因第一个位置是[CLS]
        label = model.config.id2label[prediction]
        if label.startswith("S"):
            results.append({"mention": token, "category": cate2ZH[label[2:]], "position": [pos, pos+1]})
        elif label.startswith("B"):
            entity += token
            cate = label[2:]
            postion[0] = pos
        elif label.startswith("I"):
            entity += token
            postion[1] = pos + 1
        else:
            if entity and cate:
                results.append({"mention": entity, "category": cate2ZH[cate], "position": postion})
                entity = ""
                cate = ""
                postion = [-1, -1]
    return results
            

query = "钴属于哪个行业？在线医疗属于哪个部门？哪个组织生产和销售小米8？"
print(predict_entity(query))

# [{'mention': '钴', 'category': '行业', 'position': [0, 1]}, {'mention': '在线医疗', 'category': '行业', 'position': [8, 12]}, {'mention': '小米8', 'category': '产品', 'position': [28, 31]}]

# transformers的调用方法：普通调用和pipeline调用
# 请参考：https://huggingface.co/transformers/task_summary.html


