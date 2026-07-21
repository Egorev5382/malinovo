import os
import sys
import cv2
import torch
import numpy as np
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHARS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "I", "O", "_"
]

CLASS_NAMES = {0: "plate", 1: "car", 2: "truck", 3: "bus"}
COLORS = {
    "plate": (0, 255, 255),
    "car": (0, 255, 0),
    "truck": (255, 165, 0),
    "bus": (255, 0, 255),
}
PLATE_PATTERN = re.compile(r'^[A-Z]\d{3}[A-Z]{2}\d{2,3}$')


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


def load_lprnet(model_path):
    model = LPRNet(lpr_max_len=9, class_num=len(CHARS), dropout_rate=0)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    print("LPRNet загружена")
    return model


def decode_plate(preds):
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


def read_plate(lprnet, image):
    try:
        img = cv2.resize(image, (94, 24))
        img = img.astype("float32")
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0)
        with torch.no_grad():
            preds = lprnet(tensor)
        preds = preds.cpu().detach().numpy()
        text = decode_plate(preds[0])
        if PLATE_PATTERN.match(text):
            return text
    except Exception:
        pass
    return None


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


def draw_detections(frame, detections, lprnet):
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
                plate_text = read_plate(lprnet, plate_crop)
                if plate_text:
                    label = f"car {plate_text}"
                cv2.rectangle(frame, (plate_bbox[0], plate_bbox[1]), (plate_bbox[2], plate_bbox[3]), COLORS["plate"], 2)
                break
        cv2.putText(frame, label, (x1, y1 - 8), font, font_scale, COLORS["car"], thickness)

    for bbox in detections["trucks"]:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["truck"], 2)
        label = "truck"
        for plate_bbox in detections["plates"]:
            if box_inside(plate_bbox, bbox):
                plate_crop = frame[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]
                plate_text = read_plate(lprnet, plate_crop)
                if plate_text:
                    label = f"truck {plate_text}"
                cv2.rectangle(frame, (plate_bbox[0], plate_bbox[1]), (plate_bbox[2], plate_bbox[3]), COLORS["plate"], 2)
                break
        cv2.putText(frame, label, (x1, y1 - 8), font, font_scale, COLORS["truck"], thickness)

    for bbox in detections["buses"]:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["bus"], 2)
        label = "bus"
        for plate_bbox in detections["plates"]:
            if box_inside(plate_bbox, bbox):
                plate_crop = frame[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]
                plate_text = read_plate(lprnet, plate_crop)
                if plate_text:
                    label = f"bus {plate_text}"
                cv2.rectangle(frame, (plate_bbox[0], plate_bbox[1]), (plate_bbox[2], plate_bbox[3]), COLORS["plate"], 2)
                break
        cv2.putText(frame, label, (x1, y1 - 8), font, font_scale, COLORS["bus"], thickness)

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
        plate_text = read_plate(lprnet, plate_crop)
        if plate_text:
            cv2.putText(frame, plate_text, (x1, y1 - 8), font, font_scale, COLORS["plate"], thickness)
        else:
            cv2.putText(frame, "plate?", (x1, y1 - 8), font, font_scale, (0, 0, 255), thickness)

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
    lprnet_path = os.path.join(base_dir, "models", "LPRNet.pth")

    if not os.path.exists(yolo_path):
        print(f"YOLO модель не найдена: {yolo_path}")
        return
    if not os.path.exists(lprnet_path):
        print(f"LPRNet модель не найдена: {lprnet_path}")
        return

    yolo = load_yolo(yolo_path)
    lprnet = load_lprnet(lprnet_path)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Не удалось открыть: {args.image}")
            return
        print(f"Обработка: {args.image}")
        detections = detect(yolo, frame)
        total = len(detections["plates"]) + len(detections["cars"]) + len(detections["trucks"]) + len(detections["buses"])
        print(f"Найдено: {total} (plates={len(detections['plates'])}, cars={len(detections['cars'])}, trucks={len(detections['trucks'])}, buses={len(detections['buses'])})")
        result = draw_detections(frame, detections, lprnet)
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
            result = draw_detections(frame, detections, lprnet)
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
