import torch

import classify
from dataset import CustomDataModule

if __name__ == '__main__':
    batch_size = 128
    data_module = CustomDataModule(batch_size=batch_size)
    data_module.setup()
    imgs = torch.ones(batch_size, 3, 224, 224)
    for i in range(0, batch_size, 1):
        imgs[i] = data_module.test_dataset.__getitem__(i + 128)[0]
    model = classify.ViolenceClass(gpu_id=[0], ckpt_root="C:/Users/HUAWEI/Desktop/AI‘/train_logs/",
                                   ckpt_model="resnet18_pretrain_test/version_0/checkpoints/resnet18_pretrain_test"
                                              "-epoch=20-val_loss=0.04.ckpt",
                                   batch_size=batch_size)
    ans = model.classify(imgs)
    print(ans)