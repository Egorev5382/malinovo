import os
import cv2
import time
import select
import logging
import threading
import subprocess
import numpy as np

logger = logging.getLogger(__name__)

RECONNECT_SEC = 60
STALE_TIMEOUT_SEC = 6
READ_IDLE_SEC = 3.0
OUT_W, OUT_H = 1280, 720
FRAME_SIZE = OUT_W * OUT_H * 3


class Camera:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self._frame = None
        self._capture_time = 0.0
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera-capture")
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                self._capture_loop()
            except Exception as e:
                logger.error(f"Камера: неожиданная ошибка: {e}")
            if not self._stop:
                time.sleep(1)

    def _start_ffmpeg(self):
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "1000000",
            "-probesize", "1000000",
            "-i", self.rtsp_url,
            "-vf", f"fps=5,scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "-"
        ]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=FRAME_SIZE * 2
        )

    def _capture_loop(self):
        proc = self._start_ffmpeg()
        connected_at = time.monotonic()

        deadline = time.monotonic() + 10
        buf = b""
        while not self._stop and time.monotonic() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace")[:500]
                logger.error(f"FFmpeg завершился до старта (код {proc.returncode}): {err}")
                return
            r, _, _ = select.select([proc.stdout], [], [], READ_IDLE_SEC)
            if not r:
                logger.warning("Нет данных от камеры при старте — переподключение")
                break
            chunk = proc.stdout.read(FRAME_SIZE * 2)
            if not chunk:
                time.sleep(0.1)
                continue
            buf += chunk
            if len(buf) >= FRAME_SIZE:
                break

        if len(buf) < FRAME_SIZE:
            err = b""
            if proc.stderr:
                try:
                    err = proc.stderr.read(1024)
                except Exception:
                    pass
            logger.error(f"Не удалось получить кадр: {err.decode(errors='replace')[:300]}")
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            return

        logger.info(f"Камера подключена: {OUT_W}x{OUT_H}")
        frame = np.frombuffer(buf[:FRAME_SIZE], dtype=np.uint8).reshape((OUT_H, OUT_W, 3))
        with self._lock:
            self._frame = frame
            self._capture_time = time.monotonic()

        while not self._stop:
            if time.monotonic() - connected_at > RECONNECT_SEC:
                logger.info("Профилактическое переподключение к камере")
                break
            try:
                r, _, _ = select.select([proc.stdout], [], [], READ_IDLE_SEC)
                if not r:
                    logger.warning(f"Нет данных от камеры {READ_IDLE_SEC:.0f} сек — переподключение")
                    break
                raw = proc.stdout.read(FRAME_SIZE)
            except Exception as e:
                logger.warning(f"Ошибка чтения кадра: {e}")
                break
            if not raw or len(raw) < FRAME_SIZE:
                if proc.poll() is not None:
                    err = b""
                    if proc.stderr:
                        try:
                            err = proc.stderr.read(512)
                        except Exception:
                            pass
                    logger.warning(f"FFmpeg завершился (код {proc.returncode}): {err.decode(errors='replace')[:200]}")
                else:
                    logger.warning("Неполный кадр — переподключение")
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((OUT_H, OUT_W, 3))
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
