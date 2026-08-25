import os
import cv2
import time
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|buffer_size;512|max_delay;500000|timeout;5000000"
)

RECONNECT_SEC = 60
STALE_TIMEOUT_SEC = 8
READ_FAIL_LIMIT = 2


class Camera:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self._cap = None
        self._frame = None
        self._capture_time = 0.0
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            self._capture_loop()

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error(f"Не удалось подключиться к камере: {self.rtsp_url}")
            time.sleep(2)
            return
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        connected_at = time.monotonic()
        logger.info(f"Подключено к камере: {self.rtsp_url}")

        fail_count = 0
        while not self._stop:
            if time.monotonic() - connected_at > RECONNECT_SEC:
                logger.info("Профилактическое переподключение к камере")
                break
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count >= READ_FAIL_LIMIT:
                    logger.warning("Не удалось получить кадр — переподключение к камере...")
                    break
                time.sleep(0.3)
                continue
            fail_count = 0
            with self._lock:
                self._frame = frame.copy()
                self._capture_time = time.monotonic()

        cap.release()
        self._cap = None
        if not self._stop:
            time.sleep(0.1)

    def get_frame(self):
        with self._lock:
            frame = self._frame
            capture_time = self._capture_time
        if frame is None:
            return None
        age = time.monotonic() - capture_time
        if age > STALE_TIMEOUT_SEC:
            logger.warning(f"Кадр устарел ({age:.0f} сек) — ожидание свежего")
            return None
        return frame

    def release(self):
        self._stop = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None
