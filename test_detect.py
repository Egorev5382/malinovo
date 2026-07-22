import os
import sys
import cv2
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_reader import PlateRecognizer

CLASS_NAMES = {0: "plate", 1: "car", 2: "truck", 3: "bus"}
COLORS = {
    "plate": (0, 255, 255),
    "car": (0, 255, 0),
    "truck": (255, 165, 0),
    "bus": (255, 0, 255),
}


def get_recognizer():
    basedir = os.path.dirname(os.path.abspath(__file__))
    for name in ["CRNN_int8.pth", "CRNN_fp32.pth"]:
        p = os.path.join(basedir, "models", name)
        if os.path.exists(p):
            return PlateRecognizer(p)
    return None


def load_yolo(model_path):
    local_yolo_dir = os.path.join(os.path.dirname(model_path), "yolov5")
    if not os.path.exists(local_yolo_dir):
        print("Клонирование YOLOv5 v7.0...")
        os.system(f'git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git "{local_yolo_dir}"')
    model = torch.hub.load(local_yolo_dir, "custom", path=model_path, source="local", trust_repo=True)
    model.conf = 0.25
    model.iou = 0.3
    model.to("cpu")
    print("YOLOv5 загружена")
    return model


def detect(yolo, frame):
    results = yolo([frame])
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


def box_inside(inner, outer):
    return (outer[0] <= inner[0] <= inner[2] <= outer[2] and
            outer[1] <= inner[1] <= inner[3] <= outer[3])


def draw_detections(frame, detections, recognizer):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(w, h) / 1500)
    thickness = max(1, int(min(w, h) / 400))

    for bbox in detections["cars"]:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["car"], 2)
        label = "car"
        for plate_bbox in detections["plates"]:
            if box_inside(plate_bbox, bbox):
                plate_crop = frame[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]
                if recognizer:
                    info = recognizer.read_plate(plate_crop)
                    if info:
                        label = f"car {info['text']}"
                cv2.rectangle(frame, (plate_bbox[0], plate_bbox[1]), (plate_bbox[2], plate_bbox[3]), COLORS["plate"], 2)
                break
        cv2.putText(frame, label, (x1, y2 + int(font_scale * 25)), font, font_scale, COLORS["car"], thickness)

    for bbox in detections["trucks"]:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["truck"], 2)
        label = "truck"
        for plate_bbox in detections["plates"]:
            if box_inside(plate_bbox, bbox):
                plate_crop = frame[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]
                if recognizer:
                    info = recognizer.read_plate(plate_crop)
                    if info:
                        label = f"truck {info['text']}"
                cv2.rectangle(frame, (plate_bbox[0], plate_bbox[1]), (plate_bbox[2], plate_bbox[3]), COLORS["plate"], 2)
                break
        cv2.putText(frame, label, (x1, y2 + int(font_scale * 25)), font, font_scale, COLORS["truck"], thickness)

    for bbox in detections["buses"]:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["bus"], 2)
        label = "bus"
        for plate_bbox in detections["plates"]:
            if box_inside(plate_bbox, bbox):
                plate_crop = frame[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]
                if recognizer:
                    info = recognizer.read_plate(plate_crop)
                    if info:
                        label = f"bus {info['text']}"
                cv2.rectangle(frame, (plate_bbox[0], plate_bbox[1]), (plate_bbox[2], plate_bbox[3]), COLORS["plate"], 2)
                break
        cv2.putText(frame, label, (x1, y2 + int(font_scale * 25)), font, font_scale, COLORS["bus"], thickness)

    matched_plate_bboxes = set()
    for d in [detections["cars"], detections["trucks"], detections["buses"]]:
        for bbox in d:
            for plate_bbox in detections["plates"]:
                if box_inside(plate_bbox, bbox):
                    matched_plate_bboxes.add(plate_bbox)

    for plate_bbox in detections["plates"]:
        if plate_bbox in matched_plate_bboxes:
            continue
        x1, y1, x2, y2 = plate_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["plate"], 2)
        plate_crop = frame[y1:y2, x1:x2]
        label = "plate?"
        if recognizer:
            info = recognizer.read_plate(plate_crop)
            if info:
                label = info["text"]
        cv2.putText(frame, label, (x1, y2 + int(font_scale * 25)), font, font_scale, COLORS["plate"], thickness)

    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Тест детекции машин и номеров")
    parser.add_argument("--image", "-i", type=str, help="Путь к изображению")
    parser.add_argument("--video", "-v", type=str, help="Путь к видео или URL потока")
    parser.add_argument("--camera", "-c", action="store_true", help="Веб-камера")
    parser.add_argument("--output", "-o", type=str, help="Сохранить результат")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_path = os.path.join(base_dir, "models", "YOLOS_cars.pt")

    if not os.path.exists(yolo_path):
        print(f"YOLO модель не найдена: {yolo_path}")
        return

    yolo = load_yolo(yolo_path)
    recognizer = get_recognizer()
    if recognizer:
        print("CRNN загружена")
    else:
        print("CRNN не найдена — работа без распознавания номеров")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Не удалось открыть: {args.image}")
            return
        print(f"Обработка: {args.image}")
        detections = detect(yolo, frame)
        total = len(detections["plates"]) + len(detections["cars"]) + len(detections["trucks"]) + len(detections["buses"])
        print(f"Найдено: {total} (plates={len(detections['plates'])}, cars={len(detections['cars'])}, trucks={len(detections['trucks'])}, buses={len(detections['buses'])})")
        result = draw_detections(frame, detections, recognizer)
        if args.output:
            cv2.imwrite(args.output, result)
            print(f"Сохранено: {args.output}")
        else:
            out_path = os.path.join(base_dir, "result.jpg")
            cv2.imwrite(out_path, result)
            print(f"Результат: {out_path}")
        cv2.imshow("Gate Control - Detection", result)
        print("Нажмите любую клавишу для закрытия...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.video or args.camera:
        source = 0 if args.camera else args.video
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Не удалось открыть: {source}")
            return
        print("Нажмите 'q' для выхода")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    break
                continue
            frame_count += 1
            if frame_count % 3 != 0:
                cv2.imshow("Gate Control - Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            detections = detect(yolo, frame)
            result = draw_detections(frame, detections, recognizer)
            cv2.imshow("Gate Control - Detection", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    else:
        print("Использование:")
        print("  python test_detect.py --image photo.jpg")
        print("  python test_detect.py --camera")
        print("  python test_detect.py --video video.mp4")
        print("  python test_detect.py --image photo.jpg --output result.jpg")


if __name__ == "__main__":
    main()
