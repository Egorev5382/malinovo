import os
import time
import yaml
import logging
import datetime
import socket
import cv2
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    logger.info("=== Система контроля въезда запущена ===")

    camera = Camera(config["camera"]["rtsp_url"])

    logger.info("Загрузка моделей...")
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
        photos_dir=BASE_DIR
    )

    gate = MQTTGate(
        broker=config["mqtt"]["broker"],
        port=config["mqtt"]["port"],
        topic=config["mqtt"]["topic"],
        username=config["mqtt"].get("username", ""),
        password=config["mqtt"].get("password", "")
    )
    gate.connect()

    try:
        import subprocess
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3)
        all_ips = out.stdout.strip().split()
        container_ip = all_ips[0] if all_ips else "0.0.0.0"
        # try to get host IP via gateway or host.docker.internal
        try:
            host_ip = socket.gethostbyname("host.docker.internal")
        except:
            host_ip = None
    except:
        container_ip = "0.0.0.0"
        host_ip = None
    port = config["web"]["port"]
    if host_ip:
        logger.info(f"=== Веб-интерфейс: http://{host_ip}:{port} ===")
    else:
        logger.info(f"=== Веб-интерфейс: http://{container_ip}:{port} (локально) | npx localtunnel --port {port} (с WiFi) ===")

    interval = config["camera"]["capture_interval"]
    gate_cooldown = config["gate"]["open_duration"]
    last_gate_time = 0
    empty_frame_counter = 0
    empty_frame_interval = 5

    logger.info(f"Интервал захвата: {interval} сек")
    logger.info("Ожидание транспорта...")

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                logger.warning("Нет кадра, ожидание...")
                time.sleep(interval)
                continue

            logger.info(f"Кадр захвачен: {frame.shape}")
            detections = detector.detect(frame)
            total = len(detections["plates"]) + len(detections["cars"]) + len(detections["trucks"]) + len(detections["buses"])
            logger.info(f"Детекция: {total} объектов (plates={len(detections['plates'])}, cars={len(detections['cars'])})")
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
                    engine = plate_info.get("engine", "?")
                    logger.info(f"Номер: {plate_text} | Тип: {vehicle_type} | Движок: {engine} | Точность: {plate_conf:.2f}")

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

            matched_plates = set()
            for item in matched:
                matched_plates.add(tuple(item["plate"]))

            for plate_bbox in detections["plates"]:
                if tuple(plate_bbox) in matched_plates:
                    continue

                plate_image = detector.crop_plate(frame, plate_bbox)
                plate_info = plate_reader.read_plate(plate_image)

                if not plate_info:
                    logger.info("Номер без машины не распознан")
                    continue

                plate_text = plate_info["text"]
                plate_conf = plate_info["confidence"]
                engine = plate_info.get("engine", "?")
                logger.info(f"Номер (без машины): {plate_text} | Движок: {engine} | Точность: {plate_conf:.2f}")

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

            if total == 0:
                empty_frame_counter += 1
                if empty_frame_counter >= empty_frame_interval:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_path = os.path.join(BASE_DIR, f"snap_{timestamp}.jpg")
                    cv2.imwrite(snap_path, frame)
                    logger.info(f"Скриншот: {snap_path}")
                    empty_frame_counter = 0
            else:
                empty_frame_counter = 0

            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Остановка системы...")
    finally:
        camera.release()
        gate.disconnect()
        logger.info("=== Система остановлена ===")


if __name__ == "__main__":
    main()
