import os
import cv2
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|buffer_size;512|max_delay;500000"
)

STALE_HASH_REPEATS = 3
MAX_CONNECTION_AGE_SEC = 600


class Camera:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None
        self._last_hash = None
        self._stale_count = 0
        self._connected_at = 0.0

    def connect(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            logger.error(f"Не удалось подключиться к камере: {self.rtsp_url}")
            return False
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._connected_at = time.monotonic()
        self._stale_count = 0
        self._last_hash = None
        logger.info(f"Подключено к камере: {self.rtsp_url}")
        return True

    @staticmethod
    def _frame_hash(frame) -> bytes:
        small = cv2.resize(frame, (64, 36))
        return np.asarray(small, dtype=np.uint16).tobytes()

    def _is_frozen(self, frame) -> bool:
        h = self._frame_hash(frame)
        if h == self._last_hash:
            self._stale_count += 1
        else:
            self._stale_count = 0
            self._last_hash = h
        return self._stale_count >= STALE_HASH_REPEATS

    def _reconnect(self, reason: str):
        logger.warning(f"{reason} — переподключение к камере...")
        self.connect()

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                return None

        ret, frame = self.cap.read()
        if not ret:
            self._reconnect("Не удалось получить кадр")
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

        if time.monotonic() - self._connected_at > MAX_CONNECTION_AGE_SEC:
            self._reconnect(f"Соединение старше {MAX_CONNECTION_AGE_SEC // 60} мин")
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

        if self._is_frozen(frame):
            self._reconnect(f"Поток замер ({self._stale_count} одинаковых кадров)")
            ret, frame = self.cap.read()
            if not ret:
                return None
            self._stale_count = 0

        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
