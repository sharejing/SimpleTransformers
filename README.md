# SimpleTransformers
Simple Transformers三种任务（分类、命名实体识别、语言模型微调）的代码样例，**可以切换多种预训练模型**。

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

## 3. 模型输出
* 注意Simple Transformers在训练过程中会自动连接到wandb(会提示注册)，在wandb上可以看到模型的拟合情况，训练结束后，也可以看到评估结果~
* 模型的输出文件可以在config.py中设置，根据我的config模型会输出以下文件：

|文件|描述|
|-|-|
|cache_dir|存放一些缓存文件，数据的features|
|outputs|模型输出结果，包括每一次epoch后的模型|
|preprocess.py|数据预处理方法|
|runs&wandb|上传到wandb上，用来可视化训练状况的|

## 4. 其他
* 似乎想不起来写啥了，就这样吧~
* 哦，对了，每个任务我都上传了完整的公开数据集，可以直接训练~
* 最后，真的感谢这些大佬，写了这么棒的工具。另外，才疏学浅，有错误或者不完善的地方，请批评指正！



