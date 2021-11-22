# SimpleTransformers
Simple Transformers四种任务（分类、命名实体识别、机器阅读理解、语言模型微调）的代码样例，**可以切换多种预训练模型**。

## Note
在实际的生产环境中，simpletransformers可以快速训练模型，但单条测试非常慢，通过测试发现，使用transformers来调用微调后的模型速度非常快。在每一个任务的文件夹中，test_transformers.py即为simpletransformers微调模型，transformers调用模型。如果在微调分类模型过程中，环境报如下错误：ImportError: dlopen: cannot load any more object with static TLS with torch built with gcc 5.5。请pip install scikit-learn==0.20.3 (python3.7)

## 1. 安装环境
* 参考Simple Transformers的原始github库自行安装，请参考：<https://github.com/ThilinaRajapakse/simpletransformers>
* 后记(2021/11/15)：以下是我的环境

|toolkits|version|
|-|-|
|python|3.7.11|
|torch|1.7.1+cu101|
|transformers|4.12.3|
|simpletransformers|0.63.0|
|numpy|1.19.2|
|scikit-learn|0.20.3|
|seqeval|1.2.2|
|wandb|0.12.6|

## 2. 代码介绍
* Quesstion_Answering、Classification和Named_Entity_Recognition的代码架构一样，所以只介绍一个~
* Fine_Tuning的代码非常简单，就不多做描述了。

|文件|描述|
|-|-|
|data|存放数据集的文件|
|config.py|模型的超参数以及训练参数|
|preprocess.py|数据预处理方法|
|run.sh|模型的运行脚本|
|test.py|模型的单条测试方法|
|test_transformers.py|transformers的测试方法，速度快|
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
* 哦，对了，分类和命名实体任务都有完整的公开数据集，可以直接训练~
* 最后，真的感谢这些大佬，写了这么棒的工具。另外，才疏学浅，有错误或者不完善的地方，请批评指正！


