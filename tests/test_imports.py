import importlib
import pkgutil

import pytest

import infer_cam_calibrator


def _all_modules():
    yield infer_cam_calibrator.__name__
    for info in pkgutil.walk_packages(
        infer_cam_calibrator.__path__, prefix=f"{infer_cam_calibrator.__name__}."
    ):
        yield info.name


ALL_MODULES = sorted(_all_modules())


def test_expected_modules_are_in_the_package():
    assert {
        "infer_cam_calibrator.calibration_service",
        "infer_cam_calibrator.config",
        "infer_cam_calibrator.inference",
        "infer_cam_calibrator.models.box",
        "infer_cam_calibrator.models.coord",
    } <= set(ALL_MODULES)


@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
