# AI' homework
## main.py

### 

### 接口调用

在 `main.py` 中，对输入图片的参数进行定义为 `imgs = torch.ones(batch_size, 3, 224, 224)`。
要构建 `main` 中的 `model`，使用以下参数调用 `classify.ViolenceClass` 函数：

```python
classify.ViolenceClass(
    ckpt_root="C:/Users/HUAWEI/Desktop/AI‘/train_logs/",
    ckpt_model="resnet18_pretrain_test/version_0/checkpoints/resnet18_pretrain_test-epoch=20-val_loss=0.04.ckpt"
)
 ckpt_root+ ckpt_model即为自定义的训练模型的地址。
ans = model.classify(imgs)是用mode输出模型的结果，打印出来ans
```
                
## classify.py  

### 实现接口类ViolenceClass

该函数的输入是n*3*224*224大小的PyTorch Tensor，输出是长度为n的Python列表（每个值为对应的预测类别，即整数0或1）。
先导入ViolenceClass类
再使用所需参数初始化ViolenceClass对象：
gpu_id: 要使用的GPU ID（0表示第一个GPU，-1表示CPU）。
ckpt_root: 包含模型文件的根目录。
ckpt_model: 检查点文件的名称。
最后使用classify方法对图像进行分类：imgs: 要分类的图像列表。

## dataset.py 
### 数据读入


## model.py 
### 创建模型


## train.py
### 训练模型

## test.py 
### 测试模型
