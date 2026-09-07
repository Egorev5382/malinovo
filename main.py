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
from zone import Zone
from data_dir import get_data_dir, resolve_db_path, migrate_old_data

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


def _box_center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def _box_bottom_center(box):
    return ((box[0] + box[2]) // 2, box[3])


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    data_dir = get_data_dir()
    migrate_old_data(data_dir)
    logger.info("=== Система контроля въезда запущена ===")
    logger.info(f"Хранилище данных: {data_dir}")

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
        db_path=resolve_db_path(config["database"]["path"], data_dir),
        photos_dir=os.path.join(data_dir, "photos")
    )

    if config.get("database", {}).get("clean_on_start", True):
        db.clear_logs_and_photos()
        dbg_dir = os.path.join(data_dir, "plate_debug")
        if os.path.isdir(dbg_dir):
            removed = 0
            for name in os.listdir(dbg_dir):
                try:
                    p = os.path.join(dbg_dir, name)
                    if os.path.isfile(p):
                        os.remove(p)
                        removed += 1
                except Exception:
                    pass
            if removed:
                logger.info(f"Очистка при запуске: удалено отладочных фото={removed}")

    use_ha = config.get("gate", {}).get("use_ha", False)
    if use_ha:
        from ha_gate import HAGate
        ha_cfg = config.get("homeassistant", {})
        gate = HAGate(
            entity_id=ha_cfg.get("entity_id", "switch.vorota"),
            ha_url=ha_cfg.get("ha_url") or None,
            ha_token=ha_cfg.get("ha_token") or None
        )
    else:
        gate = MQTTGate(
            broker=config["mqtt"]["broker"],
            port=config["mqtt"]["port"],
            topic=config["mqtt"]["topic"],
            username=config["mqtt"].get("username", ""),
            password=config["mqtt"].get("password", "")
        )
    gate.connect()

    host_ip = None
    try:
        import subprocess, json, urllib.request
        # 1) HA supervisor API
        token = os.environ.get("SUPERVISOR_TOKEN")
        if token:
            req = urllib.request.Request("http://supervisor/network/info")
            req.add_header("Authorization", f"Bearer {token}")
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read())
            for iface in data.get("data", {}).get("interfaces", []):
                for addr in iface.get("ipv4", {}).get("address", []):
                    ip = addr.split("/")[0]
                    if ip.startswith("192.168.") or ip.startswith("10."):
                        host_ip = ip
                        break
                if host_ip:
                    break
        # 2) host.docker.internal
        if not host_ip:
            try:
                host_ip = socket.gethostbyname("host.docker.internal")
            except:
                pass
        # 3) gateway IP
        if not host_ip:
            out = subprocess.run(["ip", "route"], capture_output=True, text=True, timeout=3)
            for line in out.stdout.splitlines():
                if "default via" in line:
                    host_ip = line.split()[2]
                    break
        # 4) container IP as fallback
        if not host_ip:
            out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3)
            host_ip = out.stdout.strip().split()[0] if out.stdout.strip() else "0.0.0.0"
    except:
        host_ip = "0.0.0.0"
    port = config["web"]["port"]
    logger.info(f"=== Веб-интерфейс: http://{host_ip}:{port} ===")

    interval = config["camera"]["capture_interval"]
    last_debug_save = 0.0
    empty_frame_counter = 0
    empty_frame_interval = 5

    zone = Zone.from_config(config.get("zone"))

    active_plates = {}
    ZONE_LEAVE_SEC = 5

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

            if time.time() - last_debug_save > 15:
                last_debug_save = time.time()
                try:
                    dbg = frame.copy()
                    for b in detections["cars"]:
                        cv2.rectangle(dbg, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 4)
                    for b in detections["trucks"]:
                        cv2.rectangle(dbg, (b[0], b[1]), (b[2], b[3]), (0, 165, 255), 4)
                    for b in detections["buses"]:
                        cv2.rectangle(dbg, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 4)
                    for b in detections["plates"]:
                        cv2.rectangle(dbg, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 4)
                    if zone is not None:
                        zone.draw(dbg)
                    debug_path = os.path.join(data_dir, "debug_frame.jpg")
                    cv2.imwrite(debug_path, dbg)
                except Exception as e:
                    logger.warning(f"Не удалось сохранить отладочный кадр: {e}")

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

                    photo_path = None
                    current_time = time.time()
                    entry = active_plates.get(plate_text)
                    if entry is None:
                        entry = {"opened": False, "logged": False, "last_seen": current_time}
                        active_plates[plate_text] = entry
                    entry["last_seen"] = current_time

                    if zone is not None:
                        h, w = frame.shape[:2]
                        arrival = zone.arrival(entry, *_box_bottom_center(vehicle_bbox), w, h)
                        arrival = arrival or zone.in_zone(*_box_center(plate_bbox), w, h)
                        if not arrival:
                            logger.info(f"Машина {plate_text} вне зоны — ожидание въезда")
                            continue

                    is_allowed = db.is_allowed(plate_text)

                    gate_opened = False
                    if is_allowed and not entry["opened"]:
                        if gate.open_gate():
                            gate_opened = True
                            entry["opened"] = True
                            logger.info(f"Ворота открыты для {plate_text} (новое появление в зоне — снова открыто)")
                        else:
                            logger.error(f"Не удалось открыть ворота для {plate_text}")
                    elif not is_allowed:
                        logger.info(f"Номер {plate_text} не в базе — доступ запрещён")

                    if not entry["logged"]:
                        photo_path = db.save_photo(frame, plate_text)
                        db.add_log(
                            plate=plate_text,
                            photo_path=photo_path,
                            allowed=is_allowed,
                            gate_opened=gate_opened,
                            confidence=plate_conf
                        )
                        entry["logged"] = True

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

                photo_path = None
                current_time = time.time()
                entry = active_plates.get(plate_text)
                if entry is None:
                    entry = {"opened": False, "logged": False, "last_seen": current_time}
                    active_plates[plate_text] = entry
                entry["last_seen"] = current_time

                if zone is not None:
                    h, w = frame.shape[:2]
                    if not zone.arrival(entry, *_box_center(plate_bbox), w, h):
                        logger.info(f"Номер {plate_text} вне зоны — ожидание въезда")
                        continue

                is_allowed = db.is_allowed(plate_text)

                gate_opened = False
                if is_allowed and not entry["opened"]:
                    if gate.open_gate():
                        gate_opened = True
                        entry["opened"] = True
                        logger.info(f"Ворота открыты для {plate_text} (новое появление в зоне — снова открыто)")
                    else:
                        logger.error(f"Не удалось открыть ворота для {plate_text}")
                elif not is_allowed:
                    logger.info(f"Номер {plate_text} не в базе — доступ запрещён")

                if not entry["logged"]:
                    photo_path = db.save_photo(frame, plate_text)
                    db.add_log(
                        plate=plate_text,
                        photo_path=photo_path,
                        allowed=is_allowed,
                        gate_opened=gate_opened,
                        confidence=plate_conf
                    )
                    entry["logged"] = True

                gate.publish_plate(plate_text, is_allowed, gate_opened)

            now = time.time()
            for p in list(active_plates):
                if now - active_plates[p]["last_seen"] > ZONE_LEAVE_SEC:
                    logger.info(f"Машина {p} покинула зону — при следующем появлении ворота откроются снова")
                    del active_plates[p]

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
