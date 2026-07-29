import os
import logging
import requests

logger = logging.getLogger(__name__)


class HAGate:
    def __init__(self, entity_id: str, ha_url: str = None, ha_token: str = None):
        self.entity_id = entity_id
        self.ha_url = (ha_url or "http://supervisor/core").rstrip("/")
        token = ha_token or os.environ.get("HA_TOKEN") or os.environ.get("SUPERVISOR_TOKEN") or ""
        self.headers = {"Authorization": f"Bearer {token.strip()}", "Content-Type": "application/json"}
        if not token.strip():
            logger.warning("HA токен не найден — HA API недоступен")

    def _call_ha(self, service: str) -> bool:
        if not self.headers.get("Authorization"):
            return False
        url = f"{self.ha_url}/api/services/switch/{service}"
        try:
            r = requests.post(url, json={"entity_id": self.entity_id}, headers=self.headers, timeout=10)
            logger.info(f"HA {service}: {self.entity_id} — {r.status_code}")
            return r.status_code in (200, 201)
        except Exception as e:
            logger.error(f"HA {service} ошибка: {e}")
            return False

    def connect(self):
        ok = bool(self.headers.get("Authorization"))
        logger.info(f"HA Gate: {'подключено' if ok else 'нет токена'} к {self.ha_url}")
        return ok

    def open_gate(self):
        ok = self._call_ha("turn_on")
        if ok:
            logger.info(f"Ворота открыты: {self.entity_id}")
        return ok

    def publish_plate(self, plate: str, allowed: bool, gate_opened: bool):
        logger.info(f"HA статус: номер={plate} разрешён={allowed} ворота={gate_opened}")

    def disconnect(self):
        logger.info("HA Gate: отключено")
