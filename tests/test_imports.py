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


_NO_CONFIG_AT_IMPORT = """
import importlib, importlib.util, pathlib, sys
from infer_cam_calibrator import config

def _refuse(*args, **kwargs):
    raise AssertionError("load_config() called at import time")

config.load_config = _refuse
for name in sys.argv[1:]:
    if name.endswith(".py"):
        spec = importlib.util.spec_from_file_location(pathlib.Path(name).stem, name)
        try:
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
        except ModuleNotFoundError as exc:  # optional third-party deps (e.g. ultralytics)
            print(f"skipped {name}: {exc}")
    else:
        importlib.import_module(name)
"""


def test_importing_does_not_load_config():
    """Importing any module must not read cam_calib.conf (the default path is the
    live install), so tests work on machines without /opt/infer_cam_calibrator."""
    import pathlib
    import subprocess
    import sys

    scripts = sorted(str(p) for p in (pathlib.Path(__file__).parents[1] / "scripts").glob("*.py"))
    subprocess.run(
        [sys.executable, "-c", _NO_CONFIG_AT_IMPORT, *ALL_MODULES, *scripts],
        check=True,
    )
