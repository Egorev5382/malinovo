import os
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = {0: "plate", 1: "car", 2: "truck", 3: "bus"}


class CarDetector:
    def __init__(self, model_path: str, conf: float = 0.5, iou: float = 0.4, device: str = "cpu"):
        self.conf = conf
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
                "git", "clone", "--depth", "1",
                "https://github.com/ultralytics/yolov5.git",
                local_yolo_dir
            ], check=True)

        self.model = torch.hub.load(local_yolo_dir, "custom", path=model_path, source="local", trust_repo=True)
        self.model.conf = conf
        self.model.iou = iou
        self.model.to(device)
        logger.info("YOLOv5 модель загружена")

    def detect(self, frame: np.ndarray) -> dict:
        self.model.conf = self.conf
        self.model.iou = self.iou
        results = self.model([frame])
        labels = results.xyxyn[0][:, -1].cpu().numpy()
        cords = results.xyxyn[0][:, :-1].cpu().numpy()

        h, w = frame.shape[:2]
        output = {"plates": [], "cars": [], "trucks": [], "buses": []}

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

        return output

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
