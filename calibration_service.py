import paho.mqtt.client as mqtt
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union, cast
from label_inference import Model
from models.box import CustomEncoder

# Configure logging
logging.basicConfig(
    filename="/opt/kiosk_fw/logs/camera_calibration_inference.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CalibrationService")

# MQTT settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_CLIENT_ID = "calibration_service"
MQTT_TOPIC_CMD = "cam_calibration/cmd/process_imgs"
MQTT_TOPIC_RESPONSE = "cam_calibration/process_imgs"


class CalibrationService:
    def __init__(self, broker: str = MQTT_BROKER, port: int = MQTT_PORT) -> None:
        self.broker: str = broker
        self.port: int = port
        self.client: mqtt.Client = mqtt.Client(client_id=MQTT_CLIENT_ID)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.connect()
        

    def connect(self) -> None:
        try:
            self.client.connect(self.broker, self.port, 6000)
            logger.info(f"Connected to MQTT broker at {self.broker}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")


    def start(self) -> None:
        """Start the MQTT client loop"""
        logger.info("Starting calibration service...")
        self.client.loop_forever()
        

    def stop(self) -> None:
        """Stop the MQTT client loop"""
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Calibration service stopped")
        

    def on_connect(self, client: mqtt.Client, userdata: Any, flags: Dict[str, int], rc: int) -> None:
        """Callback for when client connects to the broker"""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            # Subscribe to command topic
            client.subscribe(MQTT_TOPIC_CMD)
            logger.info(f"Subscribed to {MQTT_TOPIC_CMD}")
        else:
            logger.error(f"Connection failed with result code {rc}")


    def on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        """Callback for when a message is received on a subscribed topic"""
        
        logger.info(f"Message received on {msg.topic}")
        
        if msg.topic == MQTT_TOPIC_CMD:
            try:
                m = Model()
                results = m.run(logger)
                serialized_results = []
                for img_boxes in results:
                    serialized_img_boxes = []
                    for box in img_boxes:
                        serialized_img_boxes.append(box.serialize())
                    serialized_results.append(serialized_img_boxes)
                self.client.publish(
                    MQTT_TOPIC_RESPONSE, 
                    json.dumps(serialized_results),
                )
                logger.info(f"Published response to {MQTT_TOPIC_RESPONSE}")
            except ImportError as e:
                logger.error(f"Failed to import Model: {e}")
                self.client.publish(
                    MQTT_TOPIC_RESPONSE, 
                    json.dumps({"error": f"Model import failed: {str(e)}"})
                )
            except Exception as e:
                logger.error(f"Error running model: {e}")
                self.client.publish(
                    MQTT_TOPIC_RESPONSE, 
                    json.dumps({"error": f"Model execution failed: {str(e)}"})
                )
                

if __name__ == "__main__":
    service = CalibrationService()
    service.start()
    