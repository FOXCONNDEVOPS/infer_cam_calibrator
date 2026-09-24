"""
Configuration file handling for camera calibration
"""
import configparser
import os
from typing import Dict, Any, Tuple

def load_config(config_path: str = "/opt/infer_cam_calibrator/cam_calib.conf") -> Dict[str, Any]:
    """
    Load configuration from the config file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing all configuration values
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Process the configuration values
    config_dict = {
        "model_path": config.get("PATHS", "MODEL_PATH"),
        "imgs_dir": config.get("PATHS", "IMGS_DIR"),
        "class_names": {int(k): v for k, v in config.items("CLASS_NAMES")},
        "cameras": {k.lower(): int(v) for k, v in config.items("CAMERAS")},
        "conf_threshold": config.getfloat("INFERENCE", "CONF_THRESHOLD"),
        "iou_threshold": config.getfloat("INFERENCE", "IOU_THRESHOLD"),
        "input_size": (
            config.getint("INFERENCE", "INPUT_SIZE_WIDTH"),
            config.getint("INFERENCE", "INPUT_SIZE_HEIGHT")
        ),
        "random_seed": config.getint("INFERENCE", "RANDOM_SEED"),
    }
    
    # Handle optional values
    if config.has_option("VISUALIZATION", "SAVE_PATH") and config.get("VISUALIZATION", "SAVE_PATH"):
        config_dict["save_path"] = config.get("VISUALIZATION", "SAVE_PATH")
    else:
        config_dict["save_path"] = None
    
    return config_dict

def get_class_names() -> Dict[int, str]:
    """Get the class names mapping from configuration"""
    return load_config().get("class_names")

def get_cameras() -> Dict[str, int]:
    """Get the camera mapping from configuration"""
    return load_config().get("cameras")

def get_model_path() -> str:
    """Get the model path from configuration"""
    return load_config().get("model_path")

def get_images_dir() -> str:
    """Get the images directory from configuration"""
    return load_config().get("imgs_dir")

def get_input_size() -> Tuple[int, int]:
    """Get the input size from configuration"""
    return load_config().get("input_size")

def get_inference_thresholds() -> Tuple[float, float]:
    """Get the inference thresholds (confidence, IoU) from configuration"""
    config = load_config()
    return config.get("conf_threshold"), config.get("iou_threshold")
