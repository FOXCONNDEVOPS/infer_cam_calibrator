import logging
import onnxruntime
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, cast
from models.box import Box
from models.coord import Coord
import matplotlib.pyplot as plt
import random
import os
import config

# Load configuration values
CLASS_NAMES = config.get_class_names()
CAMS = config.get_cameras()

class Model:

    def run(self, logger: logging.Logger) -> List[List[Box]]:
        logger.info(f"####### Starting inference #######")
        model_path = config.get_model_path()
        session = onnxruntime.InferenceSession(model_path)
        imgs: List[List[Box]] = []
        imgs_dir = config.get_images_dir()
        for img_path in os.listdir(imgs_dir):
            if not self.is_jpg(img_path):
                continue
            logger.info(f"Inference in image {img_path}")
            full_path = os.path.join(imgs_dir, img_path)
            imgs.append(self.run_inference(session, full_path))
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
        img = img.astype(np.float32) / 255.0
        img = img[:, :, ::-1]
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        return img, cam, int(distance), (orig_height, orig_width)


    def postprocess_predictions(
        self,
        outputs: List[np.ndarray],
        orig_dimensions: Tuple[int, int],
        input_dimensions: Tuple[int, int],
        distance: int,
        cam: str,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Box]:
        """Process raw model outputs to get bounding boxes, confidence scores, and class IDs"""

        # Use configured thresholds if not provided
        if conf_threshold is None or iou_threshold is None:
            conf_default, iou_default = config.get_inference_thresholds()
            conf_threshold = conf_threshold if conf_threshold is not None else conf_default
            iou_threshold = iou_threshold if iou_threshold is not None else iou_default
        outputs = outputs[0]
        bbox = outputs[:, :4, :]       # (1, 4, 35301)
        cls_probs = outputs[:, 4:, :]   # (1, 14, 35301)
        bbox = np.transpose(bbox, (0, 2, 1))       # (1, 35301, 4)
        cls_probs = np.transpose(cls_probs, (0, 2, 1))  # (1, 35301, 14)
        class_ids = np.argmax(cls_probs, axis=-1)    # (1, 35301)
        confidences = np.max(cls_probs, axis=-1)     # (1, 35301)
        mask = confidences[0] > conf_threshold
        boxes = bbox[0][mask]
        scores = confidences[0][mask]
        labels = class_ids[0][mask]

        # Apply Non-Maximum Suppression
        results = []
        processed_boxes = []
        for box in boxes:
            x_center, y_center, width, height = box
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            processed_boxes.append([x1, y1, x2, y2])
        processed_boxes = np.array(processed_boxes)
        unique_labels = np.unique(labels)
        for class_id in unique_labels:
            class_indices = np.where(labels == class_id)[0]
            if len(class_indices) > 0:
                class_boxes = processed_boxes[class_indices]
                class_scores = scores[class_indices]
                keep_indices = cv2.dnn.NMSBoxes(
                    class_boxes.tolist(),
                    class_scores.tolist(),
                    score_threshold=conf_threshold,
                    nms_threshold=iou_threshold
                )
                if isinstance(keep_indices, np.ndarray):
                    keep_indices = keep_indices.flatten().tolist()
                for idx in keep_indices:
                    actual_idx = class_indices[idx]
                    x1, y1, x2, y2 = processed_boxes[actual_idx]
                    x1 = x1/input_dimensions[0]
                    y1 = y1/input_dimensions[1]
                    x2 = x2/input_dimensions[0]
                    y2 = y2/input_dimensions[1]
                    orig_height, orig_width = orig_dimensions
                    x1 = int(x1 * orig_width)
                    y1 = int(y1 * orig_height)
                    x2 = int(x2 * orig_width)
                    y2 = int(y2 * orig_height)
                    results.append(
                        Box(
                            coord = Coord (
                                bl = (x1, y2),  # Bottom-left: min x, max y
                                br = (x2, y2),  # Bottom-right: max x, max y
                                tl = (x1, y1),  # Top-left: min x, min y
                                tr = (x2, y1),  # Top-right: max x, min y
                            ),
                            distance = distance,
                            confidence = float(scores[actual_idx]),
                            class_id = int(labels[actual_idx]),
                            class_name = CLASS_NAMES[int(labels[actual_idx])],
                            original_size = (orig_width, orig_height),
                            cam_idx = CAMS[cam],
                        )
                    )
        return results


    def run_inference(self, session: onnxruntime.InferenceSession, image_path: str, input_size: Optional[Tuple[int, int]] = None) -> List[Box]:
        """Run complete inference pipeline"""
        if input_size is None:
            input_size = config.get_input_size()
            
        img, cam, distance, orig_dimensions = self.load_image(image_path, (input_size[1], input_size[0]))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        outputs = session.run([output_name], {input_name: img})
        results = self.postprocess_predictions(outputs, orig_dimensions, input_size, distance, cam)
        self.plot_predictions(results, image_path)
        return results
    

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