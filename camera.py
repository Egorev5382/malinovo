import os
import cv2
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|buffer_size;512|max_delay;500000|timeout;5000000"
)

STALE_HASH_REPEATS = 2
MAX_CONNECTION_AGE_SEC = 300
FREEZE_DIFF_THRESHOLD = 2.5


class Camera:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None
        self._last_small = None
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
        self._last_small = None
        logger.info(f"Подключено к камере: {self.rtsp_url}")
        return True

    @staticmethod
    def _downscale(frame) -> np.ndarray:
        return cv2.resize(frame, (32, 18)).astype(np.float32)

    def _is_frozen(self, frame) -> bool:
        small = self._downscale(frame)
        if self._last_small is not None:
            diff = float(np.mean(np.abs(small - self._last_small)))
            if diff < FREEZE_DIFF_THRESHOLD:
                self._stale_count += 1
            else:
                self._stale_count = 0
        else:
            self._stale_count = 0
        self._last_small = small
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
            self._reconnect(f"Поток замер ({self._stale_count} кадров без изменений)")
            ret, frame = self.cap.read()
            if not ret:
                return None
            self._stale_count = 0

        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
