import os
import torch
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
