import os
import shutil
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARE_ROOT = "/share"
APP_DATA_SUBDIR = "gate_system"


def get_data_dir() -> str:
    """В постоянное хранилище HA (/share) или рядом с кодом на ПК."""
    try:
        if os.path.isdir(SHARE_ROOT) and os.access(SHARE_ROOT, os.W_OK):
            data_dir = os.path.join(SHARE_ROOT, APP_DATA_SUBDIR)
            os.makedirs(data_dir, exist_ok=True)
            return data_dir
    except OSError:
        pass
    return BASE_DIR


def resolve_db_path(db_path: str, data_dir: str) -> str:
    if os.path.isabs(db_path):
        return db_path
    return os.path.join(data_dir, os.path.basename(db_path))


def migrate_old_data(data_dir: str) -> None:
    """Разовый перенос старой базы/фото из папки приложения."""
    if data_dir == BASE_DIR:
        return
    old_db = os.path.join(BASE_DIR, "gate_system.db")
    new_db = os.path.join(data_dir, "gate_system.db")
    if os.path.exists(old_db) and not os.path.exists(new_db):
        try:
            shutil.copy2(old_db, new_db)
            logger.info(f"База перенесена в постоянное хранилище: {new_db}")
        except OSError as e:
            logger.warning(f"Не удалось перенести базу: {e}")
    old_photos = os.path.join(BASE_DIR, "photos")
    new_photos = os.path.join(data_dir, "photos")
    if os.path.isdir(old_photos) and not os.path.isdir(new_photos):
        try:
            shutil.copytree(old_photos, new_photos)
            logger.info(f"Фото перенесены: {new_photos}")
        except OSError as e:
            logger.warning(f"Не удалось перенести фото: {e}")
