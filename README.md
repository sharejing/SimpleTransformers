# SimpleTransformers
Simple Transformers三种任务（分类、命名实体识别、语言模型微调）的代码样例

## 1. 安装环境
* 参考Simple Transformers的原始github库自行安装，请参考：<https://github.com/ThilinaRajapakse/simpletransformers>
* 我将环境打包成docker镜像了，可以直接拉取(需要自己提前装好docker哦~)，拉取方式：docker pull ymjing/simpletransformers:cuda9.2-torch1.6-ubuntu18.04

## 2. 代码介绍
* Classification和Named_Entity_Recognition的代码架构一样，所以只介绍一个~
* Fine_Tuning的代码非常简单，就不多做描述了。

|文件|描述|
|-|-|
|data|存放数据集的文件|
|config.py|模型的超参数以及训练参数|
|preprocess.py|数据预处理方法|
|run.sh|模型的运行脚本|
|test.py|模型的单条测试方法|
|train.py|模型训练程序|




