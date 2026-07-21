import cv2
import logging

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None

    def connect(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            logger.error(f"Не удалось подключиться к камере: {self.rtsp_url}")
            return False
        logger.info(f"Подключено к камере: {self.rtsp_url}")
        return True

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                return None
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Не удалось получить кадр, переподключение...")
            self.connect()
            ret, frame = self.cap.read()
            if not ret:
                return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
