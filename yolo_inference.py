# filepath: /opt/infer_cam_calibrator/yolo_inference.py
import logging
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, cast
from models.box import Box
from models.coord import Coord
import matplotlib.pyplot as plt
import random
import os
import config
from ultralytics import YOLO

# Load configuration values
CLASS_NAMES = config.get_class_names()
CAMS = config.get_cameras()

class Model:

    def run(self, logger: logging.Logger) -> List[List[Box]]:
        logger.info(f"####### Starting inference #######")
        model_path = config.get_model_path()
        model = YOLO(model_path)
        imgs: List[List[Box]] = []
        imgs_dir = config.get_images_dir()
        for img_path in os.listdir(imgs_dir):
            if not self.is_jpg(img_path):
                continue
            logger.info(f"Inference in image {img_path}")
            full_path = os.path.join(imgs_dir, img_path)
            imgs.append(self.run_inference(model, full_path))
        logger.info(f"###### Calibration complete ######")
        
        return imgs


    def is_jpg(self, path: str) -> bool:
        return (path.split(".")[-1] == "jpg")


    def load_image(self, image_path: str, input_size: Tuple[int, int]) -> Tuple[np.ndarray, str, int, Tuple[int, int]]:
        """Load and preprocess image for inference"""
        img = cv2.imread(str(image_path))
        image_name = image_path.split("/")[-1]
        cam = image_name[:3]
        distance = image_name[4:7]
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        orig_height, orig_width = img.shape[:2]
        img = cv2.resize(img, (input_size[1], input_size[0]))
        return img, cam, int(distance), (orig_height, orig_width)


    def postprocess_predictions(
        self,
        results: Any,
        orig_dimensions: Tuple[int, int],
        input_dimensions: Tuple[int, int],
        distance: int,
        cam: str,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Box]:
        """Process YOLO model outputs to get bounding boxes, confidence scores, and class IDs"""

        # Use configured thresholds if not provided
        if conf_threshold is None or iou_threshold is None:
            conf_default, iou_default = config.get_inference_thresholds()
            conf_threshold = conf_threshold if conf_threshold is not None else conf_default
            iou_threshold = iou_threshold if iou_threshold is not None else iou_default

        orig_height, orig_width = orig_dimensions
        processed_boxes = []
        
        for box in results[0].boxes:
            if box.conf.item() < conf_threshold:
                continue
                
            xyxy = box.xyxy[0].tolist()  # Get box in (x1, y1, x2, y2) format
            x1, y1, x2, y2 = xyxy
            
            # Normalize to original dimensions
            x1 = int(x1 * orig_width / input_dimensions[0])
            y1 = int(y1 * orig_height / input_dimensions[1])
            x2 = int(x2 * orig_width / input_dimensions[0])
            y2 = int(y2 * orig_height / input_dimensions[1])
            
            processed_boxes.append(
                Box(
                    coord = Coord(
                        bl = (x1, y2),  # Bottom-left: min x, max y
                        br = (x2, y2),  # Bottom-right: max x, max y
                        tl = (x1, y1),  # Top-left: min x, min y
                        tr = (x2, y1),  # Top-right: max x, min y
                    ),
                    distance = distance,
                    confidence = float(box.conf.item()),
                    class_id = int(box.cls.item()),
                    class_name = CLASS_NAMES[int(box.cls.item())],
                    original_size = (orig_width, orig_height),
                    cam_idx = CAMS[cam],
                )
            )
            
        return processed_boxes


    def run_inference(self, model: YOLO, image_path: str, input_size: Optional[Tuple[int, int]] = None) -> List[Box]:
        """Run complete inference pipeline"""
        if input_size is None:
            input_size = config.get_input_size()
            
        img, cam, distance, orig_dimensions = self.load_image(image_path, (input_size[1], input_size[0]))
        
        # Run inference with YOLO
        results = model.predict(img, imgsz=input_size, iou=config.get_inference_thresholds()[1], 
                                conf=config.get_inference_thresholds()[0])
        
        box_results = self.postprocess_predictions(results, orig_dimensions, input_size, distance, cam)
        self.plot_predictions(box_results, image_path)
        return box_results
    

    def plot_predictions(self, predictions: List[Box], image_path: str, input_size: Optional[Tuple[int, int]] = None, save_path: Optional[str] = None) -> None:
        """
        Run inference on an image and visualize the predictions with color-coded boxes.
        
        Args:
            predictions: List of Box objects with detection results
            image_path: Path to the input image
            input_size: Input size for the model (width, height)
            save_path: Optional path to save the annotated image
            
        Returns:
            None
        """
        if input_size is None:
            input_size = config.get_input_size()
        if save_path is None:
            save_path = config.load_config().get("save_path")
        num_classes = len(CLASS_NAMES)
        color_map: Dict[int, Tuple[int, int, int]] = {}
        random_seed = config.load_config().get("random_seed", 42)
        random.seed(random_seed)  # For reproducible colors
        for class_id in range(num_classes):
            color_map[class_id] = (
                random.randint(0, 255),  # B
                random.randint(0, 255),  # G
                random.randint(0, 255)   # R
            )
        
        # Load the original image for visualization
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Draw boxes on the image
        for box in predictions:
            class_id = box.class_id
            color = color_map[class_id]
            
            # Get box coordinates
            tl = box.coord.tl
            br = box.coord.br
            
            # Draw rectangle
            cv2.rectangle(
                image, 
                (int(tl[0]), int(tl[1])),
                (int(br[0]), int(br[1])),
                color,
                2
            )
        save_img_path = save_path + "/" + image_path.split("/")[-1]
        cv2.imwrite(save_img_path, image)
    

if __name__ == "__main__":
    logging.basicConfig(
        filename="/opt/kiosk_fw/logs/camera_calibration_inference.log",
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("CalibrationService")
    m = Model()
    results: List[List[Box]] = m.run(logger)