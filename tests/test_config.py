from pathlib import Path

from infer_cam_calibrator import config

SHIPPED_CONF = Path(__file__).resolve().parents[1] / "cam_calib.conf"


def test_shipped_config_has_expected_keys():
    cfg = config.load_config(str(SHIPPED_CONF))

    assert set(cfg) == {
        "model_path",
        "imgs_dir",
        "class_names",
        "cameras",
        "conf_threshold",
        "iou_threshold",
        "input_size",
        "random_seed",
        "save_path",
    }
    assert cfg["model_path"].endswith(".onnx")
    assert cfg["cameras"] == {"rgb": 0, "nir": 1}
    assert cfg["class_names"][13] == "nucleus"
    assert cfg["input_size"] == (2592, 2592)
