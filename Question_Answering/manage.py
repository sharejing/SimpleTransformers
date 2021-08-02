# -*- encoding: utf-8 -*-
'''
@File    :   manage.py
@Time    :   2021/07/30 13:47:29
@Author  :   Sharejing
@Contact :   yymmjing@gmail.com
@Desc    :   封装接口为Web服务
'''

from flask import Flask, request, jsonify
from config import set_args
from simpletransformers.question_answering import QuestionAnsweringModel
import json


args = set_args()
model = QuestionAnsweringModel(args.model_type, "outputs/best_model/", use_cuda=False, args=vars(args))
app = Flask(__name__)


@app.route("/mrc", methods=["POST"])
def mrc():
    data = json.loads(request.data)
    context = data["context"]
    question = data["question"]
    idx = data["idx"]

    input_data = [{
        "context": context,
        "qas": [{"question": question, "id": idx}]
    }]

    result = model.predict(input_data, n_best_size=1)

    answer = result[0][0]["answer"][0]
    idx = result[0][0]["id"]
    confidence = result[1][0]["probability"][0]
    new_result = {
        "id": idx,
        "answer": answer,
        "confidence": confidence
    }
    return jsonify({"status": 0, "message": "succeed!", "result": new_result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8088)



