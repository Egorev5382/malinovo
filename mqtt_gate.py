import json
import logging
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MQTTGate:
    def __init__(self, broker: str, port: int = 1883,
                 topic: str = "gate/open",
                 username: str = "", password: str = ""):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if username:
            self.client.username_pw_set(username, password)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.connected = False

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.connected = True
            logger.info(f"MQTT подключён к {self.broker}:{self.port}")
        else:
            logger.error(f"MQTT ошибка подключения: {rc}")

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        self.connected = False
        logger.warning("MQTT отключён")

    def connect(self):
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            logger.info("MQTT запущен")
            return True
        except Exception as e:
            logger.error(f"MQTT ошибка: {e}")
            return False

    def open_gate(self):
        if not self.connected:
            logger.warning("MQTT не подключён, попытка переподключения...")
            self.connect()
        payload = json.dumps({"action": "open"})
        result = self.client.publish(self.topic, payload)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            logger.info(f"Команда открытия ворот отправлена: {self.topic}")
            return True
        else:
            logger.error("Ошибка отправки MQTT команды")
            return False

    def publish_plate(self, plate: str, allowed: bool, gate_opened: bool):
        topic_status = self.topic.rsplit("/", 1)[0] + "/plate_detected"
        payload = json.dumps({
            "plate": plate,
            "allowed": allowed,
            "gate_opened": gate_opened
        })
        self.client.publish(topic_status, payload, retain=True)
        logger.info(f"Статус опубликован: {topic_status}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("MQTT отключён")
