import os
import cv2
import time
import logging
import threading
import subprocess
import numpy as np

logger = logging.getLogger(__name__)

RECONNECT_SEC = 60
STALE_TIMEOUT_SEC = 6
KNOWN_RESOLUTIONS = [(2880, 1616), (1920, 1080), (1280, 720), (640, 480)]


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
            "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "1000000",
            "-probesize", "1000000",
            "-i", self.rtsp_url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-r", "5",
            "-an",
            "-"
        ]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )

    def _capture_loop(self):
        proc = self._start_ffmpeg()
        connected_at = time.monotonic()
        logger.info(f"FFmpeg запущен: {self.rtsp_url}")

        width, height = None, None
        frame_size = None

        deadline = time.monotonic() + 8
        buf = b""
        while not self._stop and time.monotonic() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace")[:200]
                logger.error(f"FFmpeg завершился (код {proc.returncode}): {err}")
                time.sleep(1)
                return
            chunk = proc.stdout.read(1920 * 1080 * 3)
            if not chunk:
                time.sleep(0.1)
                continue
            buf += chunk
            for w, h in KNOWN_RESOLUTIONS:
                need = w * h * 3
                if len(buf) >= need:
                    width, height = w, h
                    frame_size = need
                    break
            if width:
                break

        if not width:
            err = proc.stderr.read().decode(errors="replace")[:300] if proc.stderr else ""
            logger.error(f"Не удалось определить разрешение камеры: {err}")
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                pass
            time.sleep(1)
            return

        logger.info(f"Камера: {width}x{height}")

        if len(buf) >= frame_size:
            first_frame = np.frombuffer(buf[:frame_size], dtype=np.uint8).reshape((height, width, 3))
            with self._lock:
                self._frame = first_frame
                self._capture_time = time.monotonic()

        while not self._stop:
            if time.monotonic() - connected_at > RECONNECT_SEC:
                logger.info("Профилактическое переподключение к камере")
                break
            raw = proc.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                if proc.poll() is not None:
                    err = proc.stderr.read().decode(errors="replace")[:200] if proc.stderr else ""
                    logger.warning(f"FFmpeg завершился (код {proc.returncode}): {err}")
                else:
                    logger.warning("Неполный кадр — переподключение")
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
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
            time.sleep(0.3)

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
