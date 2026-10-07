"""Preprocessing and box decoding of the ONNX pipeline (no model needed)."""

import functools
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from infer_cam_calibrator import config, inference

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONF = REPO_ROOT / "cam_calib.conf"

KIOSK_SHAPE = (2592, 1944)  # (height, width) of a kiosk camera image
INPUT_SHAPE = (2592, 2592)  # (height, width) fed to the model
NUM_CLASSES = 14


@pytest.fixture
def shipped_config(monkeypatch):
    """Point the config helpers at the repo's cam_calib.conf instead of /opt."""
    monkeypatch.setattr(config, "load_config", functools.partial(config.load_config, str(SHIPPED_CONF)))
    inference._class_names.cache_clear()
    inference._cams.cache_clear()
    yield
    inference._class_names.cache_clear()
    inference._cams.cache_clear()


def _raw_output(boxes_xywh, class_ids, scores):
    """A YOLO raw output (1, 4 + NUM_CLASSES, N) holding the given boxes in input pixels."""
    out = np.zeros((1, 4 + NUM_CLASSES, len(boxes_xywh)), dtype=np.float32)
    for i, (box, cls, score) in enumerate(zip(boxes_xywh, class_ids, scores)):
        out[0, :4, i] = box
        out[0, 4 + cls, i] = score
    return [out]


def _decode(outputs, orig_shape=KIOSK_SHAPE, input_shape=INPUT_SHAPE):
    model = inference.Model(logging.getLogger("test"))
    input_dimensions = (input_shape[1], input_shape[0])  # (width, height), as in cam_calib.conf
    return model.postprocess_predictions(outputs, orig_shape, input_dimensions, 100, "rgb")


# --------------------------------------------------------------------------- letterbox


def test_letterbox_params_for_kiosk_images():
    """1944x2592 into 2592x2592: no scaling, 324 px of padding on each side."""
    r, new_unpad, pads = inference.letterbox_params(KIOSK_SHAPE, INPUT_SHAPE)
    assert r == 1.0
    assert new_unpad == (1944, 2592)
    assert pads == (0, 0, 324, 324)


@pytest.mark.parametrize(
    "orig_shape, new_shape, expected",
    [
        # Values from Ultralytics 8.3.146 LetterBox(new_shape, auto=False, scaleup=True, center=True)
        ((480, 640), (640, 640), (1.0, (640, 480), (80, 80, 0, 0))),
        ((2592, 1944), (640, 640), (640 / 2592, (480, 640), (0, 0, 80, 80))),
        ((333, 500), (640, 640), (1.28, (640, 426), (107, 107, 0, 0))),
        ((375, 500), (640, 640), (1.28, (640, 480), (80, 80, 0, 0))),
        ((101, 640), (640, 640), (1.0, (640, 101), (269, 270, 0, 0))),  # odd padding
    ],
)
def test_letterbox_params_match_ultralytics(orig_shape, new_shape, expected):
    r, new_unpad, pads = inference.letterbox_params(orig_shape, new_shape)
    assert r == pytest.approx(expected[0])
    assert new_unpad == expected[1]
    assert pads == expected[2]


def test_letterbox_pads_centred_with_114_and_keeps_pixels():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(*KIOSK_SHAPE, 3), dtype=np.uint8)

    out = inference.letterbox(img, INPUT_SHAPE)

    assert out.shape == (*INPUT_SHAPE, 3)
    assert np.array_equal(out[:, 324:324 + 1944], img)
    assert (out[:, :324] == 114).all()
    assert (out[:, 324 + 1944:] == 114).all()


def test_letterbox_downscales_keeping_aspect_ratio():
    img = np.zeros((1000, 500, 3), dtype=np.uint8)

    out = inference.letterbox(img, (640, 640))

    assert out.shape == (640, 640, 3)
    assert (out[:, :160] == 114).all() and (out[:, 160 + 320:] == 114).all()
    assert (out[:, 160:160 + 320] == 0).all()


def test_load_image_letterboxes_instead_of_stretching(tmp_path):
    img = np.full((*KIOSK_SHAPE, 3), 200, dtype=np.uint8)
    path = tmp_path / "rgb_100.jpg"
    cv2.imwrite(str(path), img)

    tensor, cam, distance, orig = inference.Model(logging.getLogger("test")).load_image(str(path), INPUT_SHAPE)

    assert (cam, distance, orig) == ("rgb", 100, KIOSK_SHAPE)
    assert tensor.shape == (1, 3, *INPUT_SHAPE)
    assert tensor.dtype == np.float32
    assert np.allclose(tensor[0, :, :, :324], 114 / 255)
    assert np.allclose(tensor[0, :, :, 324:324 + 1944], 200 / 255, atol=2 / 255)


# --------------------------------------------------------------------------- box mapping


def test_unletterbox_inverts_letterbox():
    orig_shape, new_shape = (1000, 500), (640, 640)
    r, _, (top, _, left, _) = inference.letterbox_params(orig_shape, new_shape)
    original = np.array([[10.0, 20.0, 110.0, 220.0], [0.0, 0.0, 500.0, 1000.0]])
    letterboxed = original * r + [left, top, left, top]

    back = inference.unletterbox_boxes(letterboxed, orig_shape, new_shape)

    assert np.allclose(back, original)


def test_unletterbox_clips_to_the_image():
    back = inference.unletterbox_boxes(np.array([[300.0, -5.0, 2300.0, 2600.0]]), KIOSK_SHAPE, INPUT_SHAPE)
    assert back.tolist() == [[0.0, 0.0, 1944.0, 2592.0]]


def test_synthetic_box_maps_back_to_original_corners(shipped_config):
    """A box drawn at known original coordinates comes back at those coordinates."""
    x1, y1, x2, y2 = 100, 200, 160, 250  # original image pixels
    left = 324  # letterbox padding of a 1944x2592 image
    xywh = ((x1 + x2) / 2 + left, (y1 + y2) / 2, x2 - x1, y2 - y1)

    boxes = _decode(_raw_output([xywh], [3], [0.9]))

    assert len(boxes) == 1
    box = boxes[0]
    assert box.coord.tl == (x1, y1)
    assert box.coord.tr == (x2, y1)
    assert box.coord.bl == (x1, y2)
    assert box.coord.br == (x2, y2)
    assert (box.class_id, box.class_name, box.cam_idx, box.distance) == (3, "d", 0, 100)
    assert box.original_size == (1944, 2592)


def test_corners_are_rounded_not_truncated(shipped_config):
    # Edges at 10.7 / 20.6 / 50.7 / 60.6 in original pixels
    xywh = (30.7 + 324, 40.6, 40.0, 40.0)

    box = _decode(_raw_output([xywh], [0], [0.9]))[0]

    assert box.coord.tl == (11, 21)
    assert box.coord.br == (51, 61)


def test_corners_are_python_ints(shipped_config):
    """kiosk_fw's Box parser rejects anything but int coordinates; numpy ints don't serialize."""
    xywh = (500.3 + 324, 600.2, 33.3, 44.4)

    box = _decode(_raw_output([xywh], [1], [0.9]))[0]

    for corner in (box.coord.bl, box.coord.br, box.coord.tl, box.coord.tr):
        assert all(type(v) is int for v in corner)
    assert all(type(v) is int for v in box.original_size)


def test_no_detections_above_threshold(shipped_config):
    assert _decode(_raw_output([(500.0, 500.0, 40.0, 40.0)], [0], [0.1])) == []
