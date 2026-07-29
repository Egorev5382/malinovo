import os
import json
import time
import logging
import urllib.request

logger = logging.getLogger(__name__)


class HAGate:
    def __init__(self, entity_id: str, ha_url: str = None, ha_token: str = None):
        self.entity_id = entity_id
        self.ha_url = ha_url or "http://supervisor/core"
        self.token = ha_token or os.environ.get("HA_TOKEN") or os.environ.get("SUPERVISOR_TOKEN")
        if not self.token:
            logger.warning("HA токен не найден — HA API недоступен")

    def _call_ha(self, service: str) -> bool:
        if not self.token:
            return False
        url = f"{self.ha_url}/api/services/switch/{service}"
        payload = json.dumps({"entity_id": self.entity_id}).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                code = resp.status
                logger.info(f"HA {service}: {self.entity_id} — {code}")
                return code in (200, 201)
        except Exception as e:
            logger.error(f"HA {service} ошибка: {e}")
            return False

    def connect(self):
        if self.token:
            logger.info(f"HA Gate: подключено к {self.ha_url}")
            return True
        logger.warning("HA Gate: нет токена")
        return False

    def open_gate(self):
        """Импульс: ON → 0.3s → OFF"""
        if not self._call_ha("turn_on"):
            return False
        time.sleep(0.3)
        self._call_ha("turn_off")
        logger.info(f"Импульс открытия ворот: {self.entity_id}")
        return True

    def publish_plate(self, plate: str, allowed: bool, gate_opened: bool):
        logger.info(f"HA статус: номер={plate} разрешён={allowed} ворота={gate_opened}")

    def disconnect(self):
        logger.info("HA Gate: отключено")
