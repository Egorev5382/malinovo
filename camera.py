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
OUT_W, OUT_H = 1280, 720


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
            bufsize=OUT_W * OUT_H * 3 * 2
        )

    def _capture_loop(self):
        proc = self._start_ffmpeg()
        connected_at = time.monotonic()
        frame_size = OUT_W * OUT_H * 3

        deadline = time.monotonic() + 8
        buf = b""
        while not self._stop and time.monotonic() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read().decode(errors="replace")[:300]
                logger.error(f"FFmpeg завершился (код {proc.returncode}): {err}")
                time.sleep(1)
                return
            chunk = proc.stdout.read(frame_size * 2)
            if not chunk:
                time.sleep(0.1)
                continue
            buf += chunk
            if len(buf) >= frame_size:
                break

        if len(buf) < frame_size:
            err = proc.stderr.read().decode(errors="replace")[:300] if proc.stderr else ""
            logger.error(f"Не удалось получить кадр от ffmpeg: {err}")
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                pass
            time.sleep(1)
            return

        logger.info(f"Камера: {OUT_W}x{OUT_H}")
        first_frame = np.frombuffer(buf[:frame_size], dtype=np.uint8).reshape((OUT_H, OUT_W, 3))
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
