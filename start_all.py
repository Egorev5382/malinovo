import os
import sys
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def run_web():
    from web_app import app
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    logger.info(f"Веб-сервер запущен на http://0.0.0.0:{config['web']['port']}")
    app.run(host=config["web"]["host"], port=config["web"]["port"], debug=False)


def run_detector():
    from main import main
    main()


if __name__ == "__main__":
    logger.info("=== Запуск всех компонентов ===")
    web_thread = threading.Thread(target=run_web, daemon=True)
    web_thread.start()
    run_detector()
