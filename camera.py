import os
import cv2
import time
import struct
import logging
import threading
import subprocess
import numpy as np

logger = logging.getLogger(__name__)

RECONNECT_SEC = 60
STALE_TIMEOUT_SEC = 6
FRAME_HEADER = b"FRAME:"


class Camera:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self._frame = None
        self._capture_time = 0.0
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            self._capture_loop()

    def _start_ffmpeg(self):
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "500000",
            "-probesize", "50000",
            "-i", self.rtsp_url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", "scale=1280:720",
            "-r", "5",
            "-an",
            "-movflags", "frag_keyframe+empty_moov",
            "-"
        ]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1280 * 720 * 3 * 5
        )

    def _capture_loop(self):
        proc = self._start_ffmpeg()
        connected_at = time.monotonic()
        logger.info(f"FFmpeg подключено к камере: {self.rtsp_url}")

        width, height = 1280, 720
        frame_size = width * height * 3
        buffer = b""

        while not self._stop:
            if time.monotonic() - connected_at > RECONNECT_SEC:
                logger.info("Профилактическое переподключение к камере")
                break

            chunk = proc.stdout.read(frame_size - len(buffer) if len(buffer) < frame_size else frame_size)
            if not chunk:
                if proc.poll() is not None:
                    logger.warning(f"FFmpeg завершился (код {proc.returncode})")
                    break
                time.sleep(0.05)
                continue

            buffer += chunk
            if len(buffer) >= frame_size:
                frame = np.frombuffer(buffer[:frame_size], dtype=np.uint8).reshape((height, width, 3))
                buffer = buffer[frame_size:]
                with self._lock:
                    self._frame = frame
                    self._capture_time = time.monotonic()

        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        if not self._stop:
            time.sleep(0.5)

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
