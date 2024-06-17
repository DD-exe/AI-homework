from pytorch_lightning.loggers import TensorBoardLogger
import torch
from model import ViolenceClassifier


class ViolenceClass:
    def __init__(self, gpu_id, ckpt_root, ckpt_model, batch_size=128, log_name="resnet18_pretrain"):
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        ckpt_path = ckpt_root + ckpt_model
        self.logger = TensorBoardLogger("test_logs", name=log_name)

        self.model = ViolenceClassifier.load_from_checkpoint(ckpt_path)

    def classify(self, imgs):
        self.model.eval()
        with torch.no_grad():
            preds = []
            for img in imgs:
                output = self.model(img.unsqueeze(0))
                _, predicted = torch.max(output, 1)
                preds.append(predicted.item())

        return preds
