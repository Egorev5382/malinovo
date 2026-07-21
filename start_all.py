import os
import sys
import yaml
import signal
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    procs = []

    logger.info("Запуск системы контроля въезда...")

    camera_proc = subprocess.Popen([sys.executable, "main.py"])
    procs.append(("camera", camera_proc))
    logger.info("Камера + детекция запущены")

    web_proc = subprocess.Popen([sys.executable, "web_app.py"])
    procs.append(("web", web_proc))
    logger.info("Веб-сервер запущен (порт 8080)")

    def shutdown(sig, frame):
        logger.info("Остановка всех процессов...")
        for name, p in procs:
            try:
                p.terminate()
                logger.info(f"  {name} остановлен")
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("========================================")
    logger.info("  Веб: http://localhost:8080")
    logger.info("  Логин: admin | Пароль: gate2024")
    logger.info("  Ctrl+C для остановки")
    logger.info("========================================")

    while True:
        for name, p in procs:
            if p.poll() is not None:
                logger.error(f"Процесс {name} завершился, перезапуск...")
                if name == "camera":
                    new_p = subprocess.Popen([sys.executable, "main.py"])
                elif name == "web":
                    new_p = subprocess.Popen([sys.executable, "web_app.py"])
                idx = [i for i, (n, _) in enumerate(procs) if n == name][0]
                procs[idx] = (name, new_p)
        time.sleep(5)


if __name__ == "__main__":
    main()
