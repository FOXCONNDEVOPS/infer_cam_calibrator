from pathlib import Path

from infer_cam_calibrator import config

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONF = REPO_ROOT / "cam_calib.conf"


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
    assert cfg["cameras"] == {"rgb": 0, "nir": 1}
    assert cfg["class_names"][13] == "nucleus"
    assert cfg["input_size"] == (2592, 2592)


def test_shipped_config_model_path_points_at_weights():
    cfg = config.load_config(str(SHIPPED_CONF))
    model_path = Path(cfg["model_path"])

    assert model_path == Path("/opt/infer_cam_calibrator/weights") / model_path.name
    assert (REPO_ROOT / "weights" / model_path.name).is_file()
