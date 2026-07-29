import os
import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN")
SUPERVISOR_API = os.environ.get("SUPERVISOR_API", "http://supervisor")


class HAGate:
    def __init__(self, entity_id: str = "switch.vorota"):
        self.entity_id = entity_id
        self.token = SUPERVISOR_TOKEN
        if not self.token:
            logger.warning("SUPERVISOR_TOKEN не найден — HA API недоступен")

    def _call_ha(self, service: str, data: dict = None) -> bool:
        if not self.token:
            return False
        url = f"{SUPERVISOR_API}/core/api/services/switch/{service}"
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
                logger.info(f"HA {service}: {self.entity_id} — {resp.status}")
                return resp.status == 200
        except Exception as e:
            logger.error(f"HA ошибка: {e}")
            return False

    def connect(self):
        if self.token:
            logger.info("HA Gate: подключено к Supervisor API")
            return True
        logger.warning("HA Gate: нет токена")
        return False

    def open_gate(self):
        return self._call_ha("turn_on")

    def publish_plate(self, plate: str, allowed: bool, gate_opened: bool):
        logger.info(f"HA статус: номер={plate} разрешён={allowed} ворота={gate_opened}")

    def disconnect(self):
        logger.info("HA Gate: отключено")
