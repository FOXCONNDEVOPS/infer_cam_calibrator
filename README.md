# infer_cam_calibrator

Camera calibration inference service for kiosk images.

This project loads a trained detection model, runs inference on a directory of calibration images, draws annotated output images, and publishes detection results over MQTT. It is designed to run as a Linux service and currently uses the ONNX-based pipeline in [inference.py](inference.py).

## What it does

- Loads camera calibration images from a configured directory.
- Runs object detection with an ONNX model through `onnxruntime`.
- Converts detections into a structured `Box` format with corner coordinates, confidence, class name, camera index, and distance.
- Saves plotted images with drawn bounding boxes.
- Exposes the workflow through MQTT so another service can trigger processing.

## Main entry point

The production service entry point is [calibration_service.py](calibration_service.py).

Behavior:
- Connects to MQTT on `localhost:1883`
- Subscribes to `cam_calibration/cmd/process_imgs`
- When any message is received on that topic, processes all `.jpg` images in the configured image directory
- Publishes serialized results to `cam_calibration/process_imgs`

## Project layout

- [calibration_service.py](calibration_service.py) — MQTT service wrapper around the inference pipeline
- [inference.py](inference.py) — ONNX inference implementation used by the service
- [label_inference.py](label_inference.py) — alternative label-file based pipeline for working from YOLO-style `.txt` labels instead of model inference
- [yolo_inference.py](yolo_inference.py) — alternative runtime using `ultralytics.YOLO`
- [export.py](export.py) — helper script to export a YOLO `.pt` model to ONNX
- [config.py](config.py) — configuration loader for [cam_calib.conf](cam_calib.conf)
- [cam_calib.conf](cam_calib.conf) — model path, image path, classes, thresholds, camera mapping, visualization path
- [models/box.py](models/box.py) — detection result data structure
- [models/coord.py](models/coord.py) — bounding-box corner coordinate structure
- [bin/install.sh](bin/install.sh) — Ubuntu-oriented install script that creates a virtual environment and installs a systemd service

## Requirements

Core Python dependencies are listed in [requirements.txt](requirements.txt), including:
- `onnxruntime`
- `opencv-python`
- `numpy`
- `matplotlib`
- `paho-mqtt`

### Optional tools

Some helper scripts use packages that are not in [requirements.txt](requirements.txt):
- `ultralytics` — required for [yolo_inference.py](yolo_inference.py) and [export.py](export.py)

If you want to use those scripts, install `ultralytics` manually.

## Installation

### Option 1: Install as a system service

The provided installer is [bin/install.sh](bin/install.sh). It:
- must be run as `root`
- installs Python `3.12.3` using `pyenv`
- creates `.venv` in the project directory
- installs dependencies from [requirements.txt](requirements.txt)
- creates and starts a `systemd` service named `calibration-service`

It also expects:
- this repository to live at `/opt/infer_cam_calibrator`
- log output under `/opt/kiosk_fw/logs`
- `mosquitto.service` to be available on the machine

### Option 2: Manual setup

1. Create a Python virtual environment.
2. Install dependencies from [requirements.txt](requirements.txt).
3. Ensure the ONNX model exists at the configured path.
4. Ensure the input image directory and output plot directory exist.
5. Start the service by running [calibration_service.py](calibration_service.py).

## Configuration

Configuration is loaded from [cam_calib.conf](cam_calib.conf).

### Paths

- `MODEL_PATH` — ONNX model file path
- `IMGS_DIR` — directory containing calibration images
- `SAVE_PATH` — directory where annotated images are written

### Class names

The current configuration maps classes as follows:
- `0-12` → `a` through `m`
- `13` → `nucleus`

### Camera mapping

- `RGB = 0`
- `NIR = 1`

### Inference settings

- `CONF_THRESHOLD = 0.4`
- `IOU_THRESHOLD = 0.25`
- `INPUT_SIZE_WIDTH = 2592`
- `INPUT_SIZE_HEIGHT = 2592`
- `RANDOM_SEED = 42`

## Image naming expectations

There are two naming conventions in the repository:

### ONNX service / [inference.py](inference.py)
This parser expects the file name to encode:
- camera type in the first 3 characters
- distance in characters `4:7`

Example pattern:
- `rgb_100.jpg`
- `nir_250.jpg`

### Label workflow / [label_inference.py](label_inference.py)
This parser expects names in the form:
- `{uuid}-{cam}-{distance}.jpg`

Example pattern:
- `123e4567-rgb-100.jpg`

If you use both pipelines, keep in mind that their filename parsing rules are different.

## Running the service

Once configured, run the service entry point in [calibration_service.py](calibration_service.py).

At runtime the service will:
1. Wait for a message on `cam_calibration/cmd/process_imgs`
2. Read all `.jpg` images from the configured image directory
3. Run inference on each image
4. Save annotated images to `SAVE_PATH`
5. Publish results as JSON to `cam_calibration/process_imgs`

## MQTT API

### Subscribe topic

- `cam_calibration/cmd/process_imgs`

### Response topic

- `cam_calibration/process_imgs`

### Request payload

The current implementation does not inspect the payload. Any message on the command topic triggers processing.

### Response payload shape

The published response is a JSON array of images, where each image contains a list of detections:

```json
[
  [
    {
      "coord": {
        "bl": [10, 40],
        "br": [50, 40],
        "tl": [10, 20],
        "tr": [50, 20]
      },
      "confidence": 0.98,
      "distance": 100,
      "class_id": 3,
      "class_name": "d",
      "original_size": [2592, 2592],
      "cam_idx": 0
    }
  ]
]
```

## Output format

Each detection is represented by `Box.serialize()` from [models/box.py](models/box.py):
- `coord`
  - `bl`
  - `br`
  - `tl`
  - `tr`
- `confidence`
- `distance`
- `class_id`
- `class_name`
- `original_size`
- `cam_idx`

## Alternative scripts

### [label_inference.py](label_inference.py)
Use this when detections already exist as YOLO-format label files and you want to convert them into the same `Box` structure and generated plots.

Expected directory layout under configured image root:
- `images/`
- `labels/`

### [yolo_inference.py](yolo_inference.py)
Alternative implementation that uses `ultralytics.YOLO` directly instead of ONNX Runtime.

### [export.py](export.py)
Utility for exporting a YOLO `.pt` model to ONNX.

## Logs

The service writes logs to:
- `/opt/kiosk_fw/logs/camera_calibration_inference.log`
- `/opt/kiosk_fw/logs/calibration-service-output.log`
- `/opt/kiosk_fw/logs/calibration-service-error.log`

## Notes

- The service currently imports `Model` from [inference.py](inference.py), so the ONNX pipeline is the active production path.
- Annotated images are always written when `SAVE_PATH` is configured.
- Only `.jpg` files are processed.
- The code assumes a Linux deployment with MQTT and systemd available.
