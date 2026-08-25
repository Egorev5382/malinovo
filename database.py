import os
import cv2
import difflib
import datetime
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)
Base = declarative_base()


class AllowedPlate(Base):
    __tablename__ = 'allowed_plates'
    id = Column(Integer, primary_key=True)
    plate = Column(String(20), unique=True, nullable=False)
    owner = Column(String(100), default='')
    added_at = Column(DateTime, default=datetime.datetime.utcnow)


class EntryLog(Base):
    __tablename__ = 'entry_logs'
    id = Column(Integer, primary_key=True)
    plate = Column(String(20), nullable=False)
    detected_at = Column(DateTime, default=datetime.datetime.utcnow)
    photo_path = Column(String(500))
    allowed = Column(Boolean, default=False)
    gate_opened = Column(Boolean, default=False)
    confidence = Column(Float, default=0.0)


class Database:
    def __init__(self, db_path: str, photos_dir: str = "photos"):
        self.db_path = db_path
        self.photos_dir = photos_dir
        os.makedirs(photos_dir, exist_ok=True)
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"База данных инициализирована: {db_path}")

    def add_plate(self, plate: str, owner: str = "") -> bool:
        session = self.Session()
        try:
            existing = session.query(AllowedPlate).filter_by(plate=plate).first()
            if existing:
                return False
            new_plate = AllowedPlate(plate=plate, owner=owner)
            session.add(new_plate)
            session.commit()
            logger.info(f"Номер добавлен: {plate} ({owner})")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Ошибка добавления номера: {e}")
            return False
        finally:
            session.close()

    def remove_plate(self, plate: str) -> bool:
        session = self.Session()
        try:
            result = session.query(AllowedPlate).filter_by(plate=plate).delete()
            session.commit()
            if result:
                logger.info(f"Номер удалён: {plate}")
            return result > 0
        except Exception as e:
            session.rollback()
            logger.error(f"Ошибка удаления номера: {e}")
            return False
        finally:
            session.close()

    def is_allowed(self, plate: str) -> bool:
        session = self.Session()
        try:
            if session.query(AllowedPlate).filter_by(plate=plate).first() is not None:
                return True
            plates = [p.plate for p in session.query(AllowedPlate).all()]
            close = difflib.get_close_matches(plate, plates, n=1, cutoff=0.87)
            if close:
                logger.warning(
                    f"Номер {plate} распознан неточно — "
                    f"принят как {close[0]} из базы")
                return True
            return False
        finally:
            session.close()

    def get_all_plates(self) -> list:
        session = self.Session()
        try:
            plates = session.query(AllowedPlate).all()
            return [{"id": p.id, "plate": p.plate, "owner": p.owner,
                     "added_at": p.added_at.isoformat()} for p in plates]
        finally:
            session.close()

    def save_photo(self, frame, plate: str) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{plate}_{timestamp}.jpg"
        filepath = os.path.join(self.photos_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def add_log(self, plate: str, photo_path: str, allowed: bool,
                gate_opened: bool, confidence: float) -> int:
        session = self.Session()
        try:
            log = EntryLog(
                plate=plate,
                photo_path=photo_path,
                allowed=allowed,
                gate_opened=gate_opened,
                confidence=confidence
            )
            session.add(log)
            session.commit()
            logger.info(f"Лог: {plate} | разрешено: {allowed} | ворота: {gate_opened}")
            return log.id
        except Exception as e:
            session.rollback()
            logger.error(f"Ошибка записи лога: {e}")
            return -1
        finally:
            session.close()

    def get_logs(self, limit: int = 100, offset: int = 0) -> list:
        session = self.Session()
        try:
            logs = session.query(EntryLog).order_by(
                EntryLog.detected_at.desc()
            ).offset(offset).limit(limit).all()
            return [{
                "id": l.id, "plate": l.plate,
                "detected_at": l.detected_at.isoformat(),
                "photo_path": l.photo_path,
                "allowed": l.allowed,
                "gate_opened": l.gate_opened,
                "confidence": l.confidence
            } for l in logs]
        finally:
            session.close()

    def get_log_count(self) -> int:
        session = self.Session()
        try:
            return session.query(EntryLog).count()
        finally:
            session.close()
