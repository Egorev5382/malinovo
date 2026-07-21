import os
import time
import yaml
import logging
import datetime
from camera import Camera
from detector import CarDetector
from plate_reader import PlateRecognizer
from database import Database
from mqtt_gate import MQTTGate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("gate_system.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    logger.info("=== Система контроля въезда запущена ===")

    camera = Camera(config["camera"]["rtsp_url"])

    detector = CarDetector(
        model_path=config["detector"]["yolo_model"],
        conf=config["detector"]["confidence"],
        iou=config["detector"]["iou"],
        device=config["detector"]["device"]
    )

    plate_reader = PlateRecognizer(
        model_path=config["plate_reader"]["model_path"],
        device=config["detector"]["device"]
    )

    db = Database(
        db_path=config["database"]["path"],
        photos_dir=config["database"]["photos_dir"]
    )

    gate = MQTTGate(
        broker=config["mqtt"]["broker"],
        port=config["mqtt"]["port"],
        topic=config["mqtt"]["topic"],
        username=config["mqtt"].get("username", ""),
        password=config["mqtt"].get("password", "")
    )
    gate.connect()

    interval = config["camera"]["capture_interval"]
    gate_cooldown = config["gate"]["open_duration"]
    last_gate_time = 0

    logger.info(f"Интервал захвата: {interval} сек")
    logger.info("Ожидание транспорта...")

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                logger.warning("Нет кадра, ожидание...")
                time.sleep(interval)
                continue

            detections = detector.detect(frame)
            matched = detector.match_plates_to_vehicles(detections)

            if matched:
                logger.info(f"Найдено транспортных средств: {len(matched)}")
                for item in matched:
                    plate_bbox = item["plate"]
                    vehicle_bbox = item["vehicle"]
                    vehicle_type = item["type"]

                    plate_image = detector.crop_plate(frame, plate_bbox)
                    plate_info = plate_reader.read_plate(plate_image)

                    if not plate_info:
                        vehicle_image = detector.crop_vehicle(frame, vehicle_bbox)
                        plate_info = plate_reader.read_plate(vehicle_image)

                    if not plate_info:
                        logger.info("Номер не распознан, пропуск")
                        continue

                    plate_text = plate_info["text"]
                    plate_conf = plate_info["confidence"]
                    logger.info(f"Номер: {plate_text} | Тип: {vehicle_type} | Точность: {plate_conf:.2f}")

                    photo_path = db.save_photo(frame, plate_text)
                    is_allowed = db.is_allowed(plate_text)

                    gate_opened = False
                    current_time = time.time()
                    if is_allowed and (current_time - last_gate_time) > gate_cooldown:
                        if gate.open_gate():
                            gate_opened = True
                            last_gate_time = current_time
                            logger.info(f"Ворота открыты для {plate_text}")
                        else:
                            logger.error(f"Не удалось открыть ворота для {plate_text}")
                    elif not is_allowed:
                        logger.info(f"Номер {plate_text} не в базе — доступ запрещён")

                    db.add_log(
                        plate=plate_text,
                        photo_path=photo_path,
                        allowed=is_allowed,
                        gate_opened=gate_opened,
                        confidence=plate_conf
                    )

                    gate.publish_plate(plate_text, is_allowed, gate_opened)

            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Остановка системы...")
    finally:
        camera.release()
        gate.disconnect()
        logger.info("=== Система остановлена ===")


if __name__ == "__main__":
    main()
