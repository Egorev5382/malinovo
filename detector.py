import os
import torch
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "plate", 1: "car", 2: "truck", 3: "bus"}

COCO_VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}


class CarDetector:
    def __init__(self, model_path: str, conf: float = 0.5, iou: float = 0.4, device: str = "cpu"):
        self.conf = conf
        self.det_conf = min(conf, 0.3)
        self.iou = iou
        self.device = device
        logger.info(f"Загрузка YOLOv5 модели: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Модель не найдена: {model_path}")
            raise FileNotFoundError(f"YOLO модель не найдена: {model_path}")

        local_yolo_dir = os.path.join(os.path.dirname(model_path), "yolov5")
        if not os.path.exists(local_yolo_dir):
            import subprocess
            logger.info("Клонирование YOLOv5 в models/yolov5...")
            subprocess.run([
                "git", "clone", "--depth", "1", "--branch", "v7.0",
                "https://github.com/ultralytics/yolov5.git",
                local_yolo_dir
            ], check=True)

        exp_path = os.path.join(local_yolo_dir, "models", "experimental.py")
        if os.path.exists(exp_path):
            with open(exp_path, "r") as f:
                content = f.read()
            old = "ckpt = torch.load(attempt_download(w), map_location='cpu')  # load"
            new = "ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)  # load"
            if old in content:
                with open(exp_path, "w") as f:
                    f.write(content.replace(old, new))
                logger.info("YOLOv5 experimental.py patched for PyTorch 2.13+")
        self.model = torch.hub.load(local_yolo_dir, "custom", path=model_path, source="local", trust_repo=True)
        self.model.conf = conf
        self.model.iou = iou
        self.model.to(device)
        logger.info("YOLOv5 модель загружена")

        self.yolov8 = None
        try:
            from ultralytics import YOLO
            self.yolov8 = YOLO("yolov8n.pt")
            logger.info("YOLOv8n загружен (доп. детекция)")
        except Exception as e:
            logger.warning(f"YOLOv8n недоступен: {e}")

    def detect(self, frame: np.ndarray) -> dict:
        output = {"plates": [], "cars": [], "trucks": [], "buses": []}

        self.model.conf = self.det_conf
        self.model.iou = self.iou
        results = self.model([frame])
        labels = results.xyxyn[0][:, -1].cpu().numpy()
        cords = results.xyxyn[0][:, :-1].cpu().numpy()

        h, w = frame.shape[:2]
        for i in range(len(labels)):
            row = cords[i]
            x1, y1, x2, y2 = (
                int(row[0] * w), int(row[1] * h),
                int(row[2] * w), int(row[3] * h)
            )
            cls = int(labels[i])
            if cls == 0:
                output["plates"].append((x1, y1, x2, y2))
            elif cls == 1:
                output["cars"].append((x1, y1, x2, y2))
            elif cls == 2:
                output["trucks"].append((x1, y1, x2, y2))
            elif cls == 3:
                output["buses"].append((x1, y1, x2, y2))

        if not output["plates"]:
            vehicles = output["cars"] + output["trucks"] + output["buses"]
            vehicles = sorted(
                vehicles,
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                reverse=True
            )[:3]
            for (vx1, vy1, vx2, vy2) in vehicles:
                pad_x = int((vx2 - vx1) * 0.15)
                pad_y = int((vy2 - vy1) * 0.15)
                cx1 = max(0, vx1 - pad_x)
                cy1 = max(0, vy1 - pad_y)
                cx2 = min(w, vx2 + pad_x)
                cy2 = min(h, vy2 + pad_y)
                crop = frame[cy1:cy2, cx1:cx2]
                ch2, cw2 = crop.shape[:2]
                if cw2 < 80 or ch2 < 60:
                    continue
                try:
                    r3 = self.model([crop])
                    labels3 = r3.xyxyn[0][:, -1].cpu().numpy()
                    cords3 = r3.xyxyn[0][:, :-1].cpu().numpy()
                    found_crop = False
                    for i in range(len(labels3)):
                        if int(labels3[i]) != 0:
                            continue
                        row = cords3[i]
                        px1 = cx1 + int(row[0] * cw2)
                        py1 = cy1 + int(row[1] * ch2)
                        px2 = cx1 + int(row[2] * cw2)
                        py2 = cy1 + int(row[3] * ch2)
                        output["plates"].append((px1, py1, px2, py2))
                        found_crop = True
                    if found_crop:
                        logger.info(f"Номер найден на кропе машины ({cw2}x{ch2})")
                        break
                except Exception as e:
                    logger.warning(f"Детекция по кропу машины: {e}")

        if not output["plates"]:
            try:
                small = cv2.resize(frame, (w // 2, h // 2))
                r2 = self.model([small])
                labels2 = r2.xyxyn[0][:, -1].cpu().numpy()
                cords2 = r2.xyxyn[0][:, :-1].cpu().numpy()
                found_small = False
                for i in range(len(labels2)):
                    row = cords2[i]
                    x1, y1, x2, y2 = (
                        int(row[0] * w), int(row[1] * h),
                        int(row[2] * w), int(row[3] * h)
                    )
                    if int(labels2[i]) == 0:
                        output["plates"].append((x1, y1, x2, y2))
                        found_small = True
                if found_small:
                    logger.info("Номер найден на уменьшенном кадре (крупный план)")
            except Exception as e:
                logger.warning(f"Второй проход детекции: {e}")

        if self.yolov8:
            try:
                v8_results = self.yolov8(frame, conf=self.conf, iou=self.iou, verbose=False)
                for r in v8_results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls not in COCO_VEHICLE_CLASSES:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        vtype = COCO_VEHICLE_CLASSES[cls]
                        bbox = (x1, y1, x2, y2)
                        if not self._overlaps_any(bbox, output):
                            if vtype == "car":
                                output["cars"].append(bbox)
                            elif vtype == "truck":
                                output["trucks"].append(bbox)
                            elif vtype == "bus":
                                output["buses"].append(bbox)
                logger.info(f"YOLOv8: +cars={len(output['cars'])} +trucks={len(output['trucks'])} +buses={len(output['buses'])}")
            except Exception as e:
                logger.error(f"YOLOv8 ошибка: {e}")

        return output

    def _overlaps_any(self, bbox, output):
        for key in ["cars", "trucks", "buses"]:
            for existing in output[key]:
                if self._iou_box(bbox, existing) > 0.5:
                    return True
        return False

    def _iou_box(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def match_plates_to_vehicles(self, detections: dict) -> list:
        matched = []
        for plate in detections["plates"]:
            for car in detections["cars"]:
                if self._box_inside(plate, car):
                    matched.append({"plate": plate, "vehicle": car, "type": "car"})
            for truck in detections["trucks"]:
                if self._box_inside(plate, truck):
                    matched.append({"plate": plate, "vehicle": truck, "type": "truck"})
            for bus in detections["buses"]:
                if self._box_inside(plate, bus):
                    matched.append({"plate": plate, "vehicle": bus, "type": "bus"})
        return matched

    def _box_inside(self, inner, outer):
        return (outer[0] <= inner[0] <= inner[2] <= outer[2] and
                outer[1] <= inner[1] <= inner[3] <= outer[3])

    def crop_plate(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2]

    def crop_vehicle(self, frame: np.ndarray, bbox: tuple, padding: int = 15) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return frame[y1:y2, x1:x2]
