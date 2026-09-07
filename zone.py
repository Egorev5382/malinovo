import math
import logging

logger = logging.getLogger(__name__)


class Zone:
    """Зона въезда в кадре: линия или многоугольник (в относительных координатах).
    Машина 'прибыла', когда её точка пересекает линию (смена стороны)
    или входит внутрь многоугольника."""

    def __init__(self, kind="line", x1=0, y1=0, x2=0, y2=0, points=None,
                 tolerance=60, ref_w=1280, ref_h=720):
        self.ref_w = ref_w
        self.ref_h = ref_h
        self.tolerance = tolerance
        if kind == "polygon" and points:
            self.kind = "polygon"
            self.points = [(px / ref_w, py / ref_h) for px, py in points]
        else:
            self.kind = "line"
            self.x1, self.y1 = x1 / ref_w, y1 / ref_h
            self.x2, self.y2 = x2 / ref_w, y2 / ref_h

    def _line_pixels(self, frame_w, frame_h):
        return (self.x1 * frame_w, self.y1 * frame_h,
                self.x2 * frame_w, self.y2 * frame_h)

    def _poly_pixels(self, frame_w, frame_h):
        return [(px * frame_w, py * frame_h) for px, py in self.points]

    def which_side(self, x, y, frame_w, frame_h):
        x1, y1, x2, y2 = self._line_pixels(frame_w, frame_h)
        vx, vy = x2 - x1, y2 - y1
        return vx * (y - y1) - vy * (x - x1)

    def _line_distance(self, x, y, frame_w, frame_h):
        x1, y1, x2, y2 = self._line_pixels(frame_w, frame_h)
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

    def _in_poly(self, x, y, frame_w, frame_h):
        pts = self._poly_pixels(frame_w, frame_h)
        inside = False
        n = len(pts)
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and (
                    x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def in_zone(self, x, y, frame_w, frame_h):
        if self.kind == "polygon":
            return self._in_poly(x, y, frame_w, frame_h)
        return self._line_distance(x, y, frame_w, frame_h) <= self.tolerance

    def arrival(self, entry, x, y, frame_w, frame_h):
        """Обновляет трекинг номера и возвращает True, если это НОВОЕ появление в зоне."""
        if self.kind == "polygon":
            inside = self._in_poly(x, y, frame_w, frame_h)
            prev = entry.get("inside")
            entry["inside"] = inside
            if prev is None:
                return inside
            return inside and not prev
        side = 1 if self.which_side(x, y, frame_w, frame_h) >= 0 else -1
        prev = entry.get("side")
        entry["side"] = side
        if prev is None:
            return self.in_zone(x, y, frame_w, frame_h)
        return prev != side or self.in_zone(x, y, frame_w, frame_h)

    def draw(self, img, color=(255, 255, 0), thickness=3):
        import cv2
        h, w = img.shape[:2]
        if self.kind == "polygon":
            pts = [(int(x), int(y)) for x, y in self._poly_pixels(w, h)]
            cv2.polylines(img, [pts], isClosed=True, color=color,
                          thickness=thickness, lineType=cv2.LINE_AA)
            cv2.fillPoly(img, [pts], color)
            for p in pts:
                cv2.circle(img, (p[0], p[1]), 6, color, -1)
        else:
            x1, y1, x2, y2 = self._line_pixels(w, h)
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            cv2.circle(img, (int(x1), int(y1)), 8, color, -1)
            cv2.circle(img, (int(x2), int(y2)), 8, color, -1)
        return img

    @staticmethod
    def from_config(cfg):
        if not cfg or not cfg.get("enabled", False):
            logger.info("Зона не задана — работаем по всему кадру")
            return None
        ref_w = cfg.get("ref_w", 1280)
        ref_h = cfg.get("ref_h", 720)
        pts = cfg.get("points")
        if pts:
            zone = Zone(kind="polygon", points=pts,
                        tolerance=cfg.get("tolerance", 60),
                        ref_w=ref_w, ref_h=ref_h)
            logger.info(f"Зона (полигон): {pts}")
            return zone
        zone = Zone(kind="line",
                    x1=cfg.get("line_x1", 0), y1=cfg.get("line_y1", 0),
                    x2=cfg.get("line_x2", 0), y2=cfg.get("line_y2", 0),
                    tolerance=cfg.get("tolerance", 60),
                    ref_w=ref_w, ref_h=ref_h)
        logger.info(f"Зона (линия): ({zone.x1:.3f},{zone.y1:.3f})->"
                    f"({zone.x2:.3f},{zone.y2:.3f}), допуск {zone.tolerance}px")
        return zone