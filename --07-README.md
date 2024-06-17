# AI' homework
## 综述
作业项目文件包括`dataset.py` `model.py` `train.py` `test.py` `classify.py` `main.py` 。 
其中，`dataset.py` 进行数据读入，
`model.py` 用于创建模型，
运行`train.py` 训练模型，
运行`test.py` 测试模型。

`classify.py` 是定义的接口类，提供给希望使用该模型的人。只需指定已训练的模型，即可使用其输出分类结果（具体用法见下文）。

`main.py` 是接口类的一个用例，用于展示其使用方法。

`imgChanger.py` `noise.py` 是两个工具脚本，前者用于将新数据集的图片转换为模型所需的形式，后者用于在图片中加入噪声。
## 接口类classify.py  
`classify.py` 首先定义`ViolenceClass` 类，其初始化参数包括：

`gpu_id` : 要使用的GPU ID（0表示第一个GPU，-1表示CPU）

`ckpt_root` : 项目文件的根目录

`ckpt_model` : 模型文件的相对位置

其还定义了`classify` 方法,对图像进行分类:

`classify` 方法严格符合接口设计要求，即输入大小为`n*3*224*224` 的`tensor` 图像数组，输出形式为n长的分类结果数组。
## 接口用例main.py
```python
import torch
import classify
from dataset import CustomDataModule

if __name__ == '__main__':
    batch_size = 128
    data_module = CustomDataModule(batch_size=batch_size)  # 生成数据集，以进一步得到imgs数组
    data_module.setup()
    imgs = torch.ones(batch_size, 3, 224, 224)
    for i in range(0, batch_size, 1):  # 从数据集中循环取出img，得到接口设计所需的数据结构
        imgs[i] = data_module.test_dataset.__getitem__(i + 128)[0]
    model = classify.ViolenceClass(gpu_id=[0], ckpt_root="C:/Users/HUAWEI/Desktop/AI‘/train_logs/",
                                   ckpt_model="resnet18_pretrain_test/version_0/checkpoints/resnet18_pretrain_test"
                                              "-epoch=20-val_loss=0.04.ckpt",
                                   batch_size=batch_size)  # 提供模型位置，创建接口模型对象
    ans = model.classify(imgs)  # 调用classify方法，得到结果
    print(ans)  # 输出结果
```
