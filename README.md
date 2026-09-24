# infer_cam_calibrator

Camera calibration inference service for kiosk images.

This project loads a trained detection model, runs inference on a directory of calibration images, draws annotated output images, and publishes detection results over MQTT. It is designed to run as a Linux service next to kiosk_fw and currently uses the ONNX-based pipeline in [inference.py](src/infer_cam_calibrator/inference.py).

> **Merging to `master` is a production release.** kiosk_fw does not pin or tag this repository: on every firmware install and upgrade (including rollbacks) it deletes `/opt/infer_cam_calibrator`, clones `master` and runs `sudo bin/install.sh`. Whatever is on `master` is deployed to every kiosk that installs or upgrades firmware afterwards. See [Stable interface](#stable-interface-used-by-kiosk_fw).

## What it does

- Loads camera calibration images from a configured directory.
- Runs object detection with an ONNX model through `onnxruntime`.
- Converts detections into a structured `Box` format with corner coordinates, confidence, class name, camera index, and distance.
- Saves plotted images with drawn bounding boxes.
- Exposes the workflow through MQTT so another service can trigger processing.

## Main entry point

The production service entry point is [calibration_service.py](src/infer_cam_calibrator/calibration_service.py). It runs as a module from the project `.venv`:

```bash
.venv/bin/python -m infer_cam_calibrator.calibration_service
```

Behavior:
- Connects to MQTT on `localhost:1883`
- Subscribes to `cam_calibration/cmd/process_imgs`
- When any message is received on that topic, processes all `.jpg` images in the configured image directory
- Publishes serialized results to `cam_calibration/process_imgs`

## Project layout

The project is a uv-managed package with a src layout. Service code is imported as `from infer_cam_calibrator.<module> import ...` (absolute imports only, no `sys.path` changes).

- [pyproject.toml](pyproject.toml) / [uv.lock](uv.lock) — dependencies (exact pins), dependency groups and the pinned uv version
- [.python-version](.python-version) — Python `3.12.3`, installed and managed by uv
- [src/infer_cam_calibrator/](src/infer_cam_calibrator/) — the service package
  - [calibration_service.py](src/infer_cam_calibrator/calibration_service.py) — MQTT service wrapper around the inference pipeline
  - [inference.py](src/infer_cam_calibrator/inference.py) — ONNX inference implementation used by the service
  - [config.py](src/infer_cam_calibrator/config.py) — configuration loader for [cam_calib.conf](cam_calib.conf)
  - [models/box.py](src/infer_cam_calibrator/models/box.py) — detection result data structure
  - [models/coord.py](src/infer_cam_calibrator/models/coord.py) — bounding-box corner coordinate structure
- [scripts/](scripts/) — offline tools, not part of the service package
  - [label_inference.py](scripts/label_inference.py) — alternative label-file based pipeline for working from YOLO-style `.txt` labels instead of model inference
  - [yolo_inference.py](scripts/yolo_inference.py) — alternative runtime using `ultralytics.YOLO`
  - [export.py](scripts/export.py) — helper script to export a YOLO `.pt` model to ONNX
- [weights/](weights/) — the ONNX model (`cam_calibrator_1_1_9.onnx`)
- [cam_calib.conf](cam_calib.conf) — model path, image path, classes, thresholds, camera mapping, visualization path
- [tests/](tests/) — pytest tests (imports, import style, config, `bin/install.sh` invariants)
- [bin/install.sh](bin/install.sh) — kiosk installer: bootstraps uv, syncs `.venv` and installs the systemd service

## Requirements

Dependencies are declared in [pyproject.toml](pyproject.toml) and locked in [uv.lock](uv.lock). Core runtime dependencies include:
- `onnxruntime`
- `opencv-python`
- `numpy`
- `matplotlib`
- `paho-mqtt`

Dependency groups (never installed on kiosks):
- `dev` — `pytest`
- `train` — `ultralytics`, required for [yolo_inference.py](scripts/yolo_inference.py) and [export.py](scripts/export.py)

You only need [uv](https://docs.astral.sh/uv/). uv installs Python `3.12.3` itself (see [.python-version](.python-version)); pyenv is not used. The uv version is pinned in `[tool.uv] required-version` in [pyproject.toml](pyproject.toml), so use that release.

## Development

```bash
uv sync                  # create .venv with runtime + dev dependencies, project installed editable
uv run pytest            # run the tests (works from anywhere in the repo)
uv sync --group train    # also install ultralytics for the offline tools in scripts/
uv run python scripts/export.py
```

`uv sync` installs the package in editable mode, so edits under `src/` take effect without reinstalling. `pip install -e .` also works through the `uv_build` backend.

Add or change dependencies with `uv add` / `uv remove` and commit the updated [uv.lock](uv.lock): kiosks install exactly what is locked.

### Model and config

- The ONNX model lives in [weights/](weights/). `MODEL_PATH` in [cam_calib.conf](cam_calib.conf) points to it.
- [cam_calib.conf](cam_calib.conf) stays at the repository root. The config loader reads `/opt/infer_cam_calibrator/cam_calib.conf` by default (the absolute path of the kiosk install), so code that calls `load_config()` without a path reads the installed copy, not your checkout. Pass a path explicitly (as the tests do) to use another file.

## Installation

### Option 1: Install as a system service (kiosk)

This is how kiosk_fw installs it, and the only supported way to install on a kiosk:

```bash
sudo rm -rf /opt/infer_cam_calibrator
sudo git clone <repo-url> /opt/infer_cam_calibrator
cd /opt/infer_cam_calibrator
sudo bin/install.sh
```

The installer [bin/install.sh](bin/install.sh):
- must be run as `root`
- creates `/opt/kiosk_fw/logs` if missing
- installs the uv release pinned in [pyproject.toml](pyproject.toml) into `/opt/infer_cam_calibrator/.tools/uv`, verified by sha256 (it never uses a uv found on `PATH` or kiosk_fw's uv), unless that version is already there
- runs `uv sync --frozen --no-dev`, which installs Python `3.12.3` in root's default uv location and creates `.venv` exactly from [uv.lock](uv.lock)
- creates, enables and (re)starts a `systemd` service named `calibration-service`

It also expects:
- this repository to live at `/opt/infer_cam_calibrator`
- an x86_64 machine (the pinned uv asset is `x86_64-unknown-linux-gnu`)
- internet access (uv release from GitHub, Python and packages)
- `mosquitto.service` to be available on the machine

It does not touch `/root/.pyenv` left by older installs.

### Option 2: Manual setup

1. Run `uv sync --no-dev`.
2. Ensure the ONNX model exists at the configured path.
3. Ensure the input image directory and output plot directory exist.
4. Start the service with `.venv/bin/python -m infer_cam_calibrator.calibration_service`.

## Stable interface used by kiosk_fw

Every kiosk_fw version in the field, old and new, installs this repository from `master` the same way. Anything below is a contract with firmware that can no longer be changed, so **do not change it**, even if it looks unused here:

- **Invocation:** `sudo bin/install.sh` with no arguments, as root, on a fresh clone of `master` at `/opt/infer_cam_calibrator`. Do not add required arguments, environment variables or prompts.
- **Self-contained:** the script installs everything it needs itself, including its own pinned uv. It must not rely on kiosk_fw's uv, a uv on `PATH` or pyenv.
- **Service:** it creates, enables and starts the systemd unit `calibration-service` (running as root, `WorkingDirectory=/opt/infer_cam_calibrator`, `ExecStart=.venv/bin/python -m infer_cam_calibrator.calibration_service`).
- **Logs:** the service logs to `/opt/kiosk_fw/logs` (`calibration-service-output.log`, `calibration-service-error.log`, `camera_calibration_inference.log`), and the script creates that folder if missing.
- **MQTT topics:** command `cam_calibration/cmd/process_imgs`, reply `cam_calibration/process_imgs`, with the payloads described in [MQTT API](#mqtt-api).
- **Shared folder:** images are read from and plots written to `/opt/kiosk_fw/configuration/config_files/camera_calibration/` (`images/` and `plots/`, see [cam_calib.conf](cam_calib.conf)).

Remember that merging to `master` deploys to every kiosk that installs or upgrades firmware afterwards, so validate changes to this interface (and anything else) on a dev kiosk before merging.

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

### ONNX service / [inference.py](src/infer_cam_calibrator/inference.py)
This parser expects the file name to encode:
- camera type in the first 3 characters
- distance in characters `4:7`

Example pattern:
- `rgb_100.jpg`
- `nir_250.jpg`

### Label workflow / [label_inference.py](scripts/label_inference.py)
This parser expects names in the form:
- `{uuid}-{cam}-{distance}.jpg`

Example pattern:
- `123e4567-rgb-100.jpg`

If you use both pipelines, keep in mind that their filename parsing rules are different.

## Running the service

On a kiosk the service runs as `calibration-service` (`systemctl status calibration-service`). Manually, run `uv run python -m infer_cam_calibrator.calibration_service`.

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

Each detection is represented by `Box.serialize()` from [models/box.py](src/infer_cam_calibrator/models/box.py):
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

These offline tools live in [scripts/](scripts/) and import the package with absolute imports. Install their dependencies with `uv sync --group train`.

### [label_inference.py](scripts/label_inference.py)
Use this when detections already exist as YOLO-format label files and you want to convert them into the same `Box` structure and generated plots.

Expected directory layout under configured image root:
- `images/`
- `labels/`

### [yolo_inference.py](scripts/yolo_inference.py)
Alternative implementation that uses `ultralytics.YOLO` directly instead of ONNX Runtime.

### [export.py](scripts/export.py)
Utility for exporting a YOLO `.pt` model to ONNX.

## Logs

The service writes logs to:
- `/opt/kiosk_fw/logs/camera_calibration_inference.log`
- `/opt/kiosk_fw/logs/calibration-service-output.log`
- `/opt/kiosk_fw/logs/calibration-service-error.log`

## Notes

- The service currently imports `Model` from [inference.py](src/infer_cam_calibrator/inference.py), so the ONNX pipeline is the active production path.
- Annotated images are always written when `SAVE_PATH` is configured.
- Only `.jpg` files are processed.
- The code assumes a Linux deployment with MQTT and systemd available.
