import logging
import os
import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional, Union, cast
from models.box import Box
from models.coord import Coord
import config

# Load configuration values
CLASS_NAMES = config.get_class_names()
CAMS = config.get_cameras()

class Model:
    
    def run(self, logger: logging.Logger) -> List[List[Box]]:
        logger.info(f"####### Starting label-based inference #######")
        imgs: List[List[Box]] = []
        # imgs_dir = config.get_images_dir()
        # labels_dir = os.path.join(os.path.dirname(imgs_dir), "labels")
        imgs_dir = os.path.join(config.get_images_dir(), "images")
        labels_dir = os.path.join(config.get_images_dir(), "labels")
        
        for img_path in os.listdir(imgs_dir):
            if not self.is_jpg(img_path):
                continue
            logger.info(f"Processing labels for image {img_path}")
            full_img_path = os.path.join(imgs_dir, img_path)
            label_path = os.path.join(labels_dir, img_path.replace(".jpg", ".txt"))
            if not os.path.exists(label_path):
                logger.warning(f"No label file found for {img_path}")
                continue
            imgs.append(self.process_label_file(label_path, full_img_path))
            
        logger.info(f"###### Label processing complete ######")
        return imgs
    
    def is_jpg(self, path: str) -> bool:
        return (path.split(".")[-1] == "jpg")
    
    def load_image_dimensions(self, image_path: str) -> Tuple[str, int, Tuple[int, int]]:
        """Load image metadata for label processing"""
        img = cv2.imread(str(image_path))
        image_name = image_path.split("/")[-1]
        
        # Extract camera type and distance from image name
        # Image name format: [uuid]-[cam_type]-[distance].jpg
        parts = image_name.split("-")
        cam = parts[1]  # nir or rgb
        distance = parts[2].split(".")[0]  # extract distance and remove extension
        
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        orig_height, orig_width = img.shape[:2]
        
        return cam, int(distance), (orig_height, orig_width)

    def process_label_file(self, label_path: str, image_path: str) -> List[Box]:
        """Process label file to extract detection data"""
        cam, distance, image_dims = self.load_image_dimensions(image_path)
        orig_height, orig_width = image_dims
        
        results: List[Box] = []
        
        with open(label_path, "r") as f:
            for line in f:
                if line.startswith("//") or line.strip() == "":
                    continue
                    
                parts = line.strip().split(" ")
                if len(parts) != 5:
                    continue
                    
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert from normalized coordinates to pixel coordinates
                x1 = int((x_center - width/2) * orig_width)
                y1 = int((y_center - height/2) * orig_height)
                x2 = int((x_center + width/2) * orig_width)
                y2 = int((y_center + height/2) * orig_height)
                
                # Use class_id directly from label, map to class name
                class_name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")
                
                results.append(
                    Box(
                        coord = Coord(
                            bl = (x1, y2),  # Bottom-left: min x, max y
                            br = (x2, y2),  # Bottom-right: max x, max y
                            tl = (x1, y1),  # Top-left: min x, min y
                            tr = (x2, y1),  # Top-right: max x, min y
                        ),
                        distance = distance,
                        confidence = 1.0,  # Labels don't have confidence values
                        class_id = class_id,
                        class_name = class_name,
                        original_size = (orig_width, orig_height),
                        cam_idx = CAMS[cam],
                    )
                )
                
        self.plot_predictions(results, image_path)
        return results
    
    def plot_predictions(self, predictions: List[Box], image_path: str, input_size: Optional[Tuple[int, int]] = None, save_path: Optional[str] = None) -> None:
        """
        Visualize label predictions with color-coded boxes.
        
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

    def run_inference(self, session: Any, image_path: str, input_size: Optional[Tuple[int, int]] = None) -> List[Box]:
        """Mirror of run_inference from original Model class"""
        if input_size is None:
            input_size = config.get_input_size()
            
        # For label-based inference, we don't need a session
        label_path = image_path.replace(".jpg", ".txt")
        label_path = label_path.replace("/images/", "/labels/")
        
        if not os.path.exists(label_path):
            return []
            
        return self.process_label_file(label_path, image_path)
    
    def postprocess_predictions(
        self,
        outputs: List[Any],
        orig_dimensions: Tuple[int, int],
        input_dimensions: Tuple[int, int],
        distance: int,
        cam: str,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Box]:
        """Mirrored method for API compatibility, not used in label-based inference"""
        # This method is included for interface compatibility but not used
        # since we directly process labels in process_label_file
        return []

if __name__ == "__main__":
    logging.basicConfig(
        filename="/opt/kiosk_fw/logs/camera_calibration_label_inference.log",
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("LabelInferenceService")
    m = Model()
    results: List[List[Box]] = m.run(logger)