import os
import re
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

CHARS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "I", "O", "_"
]


class SmallBasicBlock(torch.nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(torch.nn.Module):
    def __init__(self, lpr_max_len=9, class_num=37, dropout_rate=0):
        super().__init__()
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            SmallBasicBlock(64, 128),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            SmallBasicBlock(64, 256),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            SmallBasicBlock(256, 256),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Conv2d(64, 256, kernel_size=(1, 4), stride=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Conv2d(256, class_num, kernel_size=(13, 1), stride=1),
            torch.nn.BatchNorm2d(class_num),
            torch.nn.ReLU(),
        )
        self.container = torch.nn.Sequential(
            torch.nn.Conv2d(448 + class_num, class_num, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        keep_features = []
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)
        global_context = []
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = torch.nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = torch.nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)
        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)
        return logits


class PlateRecognizer:
    def __init__(self, model_path: str, device: str = "cpu", max_len: int = 9):
        self.device = device
        self.max_len = max_len
        self.model = LPRNet(lpr_max_len=max_len, class_num=len(CHARS), dropout_rate=0)
        self.model.to(torch.device(device))

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"LPRNet модель загружена: {model_path}")
        else:
            logger.error(f"LPRNet веса не найдены: {model_path}")
            raise FileNotFoundError(f"Веса не найдены: {model_path}")

        self.model.eval()
        self.plate_pattern = re.compile(r'^[A-Z]\d{3}[A-Z]{2}\d{2,3}$')

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        import cv2
        img = cv2.resize(image, (94, 24))
        img = img.astype("float32")
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.unsqueeze(0)
        return tensor

    def decode(self, preds: np.ndarray) -> str:
        label = ""
        preds_label = []
        for j in range(preds.shape[1]):
            preds_label.append(np.argmax(preds[:, j], axis=0))
        pre_c = preds_label[0]
        if pre_c != len(CHARS) - 1:
            label += CHARS[pre_c]
        for c in preds_label:
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            label += CHARS[c]
            pre_c = c
        return label

    def is_valid_plate(self, text: str) -> bool:
        return bool(self.plate_pattern.match(text))

    def read_plate(self, image: np.ndarray) -> dict:
        if image is None or image.size == 0:
            return None
        try:
            tensor = self.preprocess(image)
            with torch.no_grad():
                preds = self.model(tensor)
            preds = preds.cpu().detach().numpy()
            text = self.decode(preds[0])
            if self.is_valid_plate(text):
                return {"text": text, "confidence": 0.9}
            return None
        except Exception as e:
            logger.error(f"Ошибка распознавания номера: {e}")
            return None
