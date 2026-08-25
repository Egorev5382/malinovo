import os
import re
import cv2
import torch
import torch.nn as nn
import numpy as np
import logging
from torchvision import transforms
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import QConfigMapping

logger = logging.getLogger(__name__)

OCR_ALPHABET = "0123456789ABCEHKMOPTXY"
PLATE_PATTERN = re.compile(r'^[A-Z]\d{3}[A-Z]{2}\d{2,3}$')


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.rnn = nn.LSTM(512 * 2, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        batch, channels, height, width = x.size()
        x = x.reshape(batch, channels * height, width)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        x = x.permute(1, 0, 2)
        x = nn.functional.log_softmax(x, dim=2)
        return x


class PlateRecognizer:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        num_classes = len(OCR_ALPHABET) + 1

        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Grayscale(),
            transforms.Resize((32, 128)),
            transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.int_to_char = {i + 1: char for i, char in enumerate(OCR_ALPHABET)}
        self.int_to_char[0] = ""

        if os.path.exists(model_path):
            is_int8 = "int8" in model_path or "quant" in model_path
            if is_int8:
                try:
                    torch.backends.quantized.engine = "qnnpack"
                    model_fp32 = CRNN(num_classes).eval()
                    qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.get_default_qconfig("qnnpack"))
                    example_inputs = (torch.randn(1, 1, 32, 128),)
                    model_prepared = quantize_fx.prepare_fx(model_fp32, qconfig_mapping, example_inputs)
                    self.model = quantize_fx.convert_fx(model_prepared)
                    self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
                    logger.info(f"CRNN INT8 загружена: {model_path}")
                except Exception as e:
                    logger.warning(f"INT8 загрузка не удалась ({e}), пробую FP32...")
                    self.model = CRNN(num_classes).eval()
                    fp32_path = model_path.replace("_int8", "_fp32")
                    if os.path.exists(fp32_path):
                        self.model.load_state_dict(torch.load(fp32_path, map_location="cpu", weights_only=True))
                        logger.info(f"CRNN FP32 загружена (fallback): {fp32_path}")
                    else:
                        raise
            else:
                self.model = CRNN(num_classes).eval()
                self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                logger.info(f"CRNN FP32 загружена: {model_path}")
        else:
            logger.error(f"CRNN веса не найдены: {model_path}")
            raise FileNotFoundError(f"КРНН модель не найдена: {model_path}")

        self.ocr = None
        try:
            import easyocr
            self.ocr = easyocr.Reader(["ru", "en"], gpu=False)
            logger.info("EasyOCR загружен (fallback)")
        except Exception as e:
            logger.warning(f"EasyOCR недоступен: {e}")

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _four_point_transform(self, image, pts):
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        if maxWidth <= 0 or maxHeight <= 0:
            return image
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def _preprocess_plate(self, plate_image):
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return plate_image
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours[:5]:
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0 or w / float(h) < 1.5:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                warped = self._four_point_transform(plate_image, approx.reshape(4, 2))
                wh, ww = warped.shape[:2]
                if ww > 0 and wh > 0 and ww / float(wh) >= 1.5:
                    return warped
        return plate_image

    def _save_debug_crop(self, plate_image):
        try:
            from data_dir import get_data_dir
            dbg_dir = os.path.join(get_data_dir(), "plate_debug")
            os.makedirs(dbg_dir, exist_ok=True)
            cv2.imwrite(os.path.join(dbg_dir, "crop_raw.jpg"), plate_image)
            cv2.imwrite(os.path.join(dbg_dir, "crop_preprocessed.jpg"),
                        self._preprocess_plate(plate_image))
        except Exception:
            pass

    @torch.no_grad()
    def _decode_crnn(self, plate_image):
        tensor = self.transform(plate_image).unsqueeze(0).to(self.device)
        preds = self.model(tensor)
        preds = preds.permute(1, 0, 2).argmax(dim=2)[0]
        decoded_seq = []
        last_char_idx = 0
        for char_idx in preds:
            char_idx = char_idx.item()
            if char_idx != 0 and char_idx != last_char_idx:
                decoded_seq.append(self.int_to_char.get(char_idx, ""))
            last_char_idx = char_idx
        return "".join(decoded_seq)

    @torch.no_grad()
    def _recognize_crnn(self, plate_image):
        try:
            candidates = []
            try:
                candidates.append(self._preprocess_plate(plate_image))
            except Exception:
                pass
            candidates.append(plate_image)

            raw_texts = []
            for variant in candidates:
                if variant is None or variant.size == 0:
                    continue
                text = self._decode_crnn(variant)
                if not text:
                    continue
                raw_texts.append(text)
                if PLATE_PATTERN.match(text):
                    return text
            if raw_texts:
                logger.info(f"CRNN сырой текст (не прошёл шаблон): {raw_texts}")
        except Exception as e:
            logger.error(f"CRNN ошибка: {e}")
        return None

    def read_plate(self, image):
        if image is None or image.size == 0:
            return None

        self._save_debug_crop(image)

        text = self._recognize_crnn(image)
        if text:
            logger.info(f"CRNN: {text}")
            return {"text": text, "confidence": 0.95, "engine": "CRNN"}

        if self.ocr:
            try:
                variants = [image]
                try:
                    pre = self._preprocess_plate(image)
                    if pre is not image and pre.size > 0:
                        variants.append(pre)
                except Exception:
                    pass
                for variant in variants:
                    results = self.ocr.readtext(variant)
                    for (_, ocr_text, conf) in results:
                        clean = re.sub(r'[^A-Za-z0-9]', "", ocr_text.upper())
                        if PLATE_PATTERN.match(clean):
                            logger.info(f"EasyOCR: {clean}")
                            return {"text": clean, "confidence": conf, "engine": "EasyOCR"}
            except Exception as e:
                logger.error(f"EasyOCR ошибка: {e}")

        return None
