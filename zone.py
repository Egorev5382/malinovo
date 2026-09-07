import math
import logging

logger = logging.getLogger(__name__)


class ZoneLine:
    """Линия въезда в кадре. Машина 'в зоне', когда её точка пересекает линию
    (расстояние до отрезка не больше tolerance пикселей)."""

    def __init__(self, x1, y1, x2, y2, tolerance=60, ref_w=1280, ref_h=720):
        self.x1 = x1 / ref_w
        self.y1 = y1 / ref_h
        self.x2 = x2 / ref_w
        self.y2 = y2 / ref_h
        self.tolerance = tolerance

    def to_pixels(self, frame_w, frame_h):
        return (self.x1 * frame_w, self.y1 * frame_h,
                self.x2 * frame_w, self.y2 * frame_h)

    def distance(self, x, y, frame_w, frame_h):
        x1, y1, x2, y2 = self.to_pixels(frame_w, frame_h)
        vx, vy = x2 - x1, y2 - y1
        wx, wy = x - x1, y - y1
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return math.hypot(x - x1, y - y1)
        c2 = vx * vx + vy * vy
        if c2 <= c1:
            return math.hypot(x - x2, y - y2)
        t = c1 / c2
        return math.hypot(x - (x1 + t * vx), y - (y1 + t * vy))

    def in_zone(self, x, y, frame_w, frame_h, tol=None):
        return self.distance(x, y, frame_w, frame_h) <= (tol or self.tolerance)

    def draw(self, img, color=(255, 255, 0), thickness=3):
        import cv2
        h, w = img.shape[:2]
        x1, y1, x2, y2 = self.to_pixels(w, h)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.circle(img, (int(x1), int(y1)), 8, color, -1)
        cv2.circle(img, (int(x2), int(y2)), 8, color, -1)
        return img

    @staticmethod
    def from_config(cfg):
        if not cfg or not cfg.get("enabled", False):
            logger.info("Зона не задана — работаем по всему кадру")
            return None
        zone = ZoneLine(
            cfg.get("line_x1", 0), cfg.get("line_y1", 0),
            cfg.get("line_x2", 0), cfg.get("line_y2", 0),
            tolerance=cfg.get("tolerance", 60),
            ref_w=cfg.get("ref_w", 1280),
            ref_h=cfg.get("ref_h", 720),
        )
        logger.info(f"Зона (линия): ({zone.x1:.3f},{zone.y1:.3f})->"
                    f"({zone.x2:.3f},{zone.y2:.3f}), допуск {zone.tolerance}px")
        return zone