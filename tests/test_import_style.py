"""Guard against the old flat layout's import patterns coming back."""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_DIRS = [REPO_ROOT / "src" / "infer_cam_calibrator", REPO_ROOT / "scripts", REPO_ROOT / "tests"]
OLD_TOP_LEVEL_NAMES = {"models", "inference", "config", "calibration_service"}

PY_FILES = sorted(p for d in SCANNED_DIRS for p in d.rglob("*.py"))


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in OLD_TOP_LEVEL_NAMES:
                    found.append(f"line {node.lineno}: bare 'import {alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                found.append(f"line {node.lineno}: relative import")
            elif node.module and node.module.split(".")[0] in OLD_TOP_LEVEL_NAMES:
                found.append(f"line {node.lineno}: bare 'from {node.module} import ...'")
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in {"insert", "append"}
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "path"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "sys"
        ):
            found.append(f"line {node.lineno}: sys.path.{node.attr}")
    return found


def test_scans_the_package_scripts_and_tests():
    scanned = {p.relative_to(REPO_ROOT).parts[0] for p in PY_FILES}
    assert scanned == {"src", "scripts", "tests"}


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_old_import_patterns(path):
    assert _violations(path) == []


@pytest.mark.parametrize(
    "source",
    [
        "import config",
        "import models.box",
        "from models.box import Box",
        "from inference import Model",
        "from calibration_service import CalibrationService",
        "from . import config",
        "import sys\nsys.path.insert(0, '.')",
        "import sys\nsys.path.append('.')",
    ],
)
def test_detects_old_import_patterns(tmp_path, source):
    bad = tmp_path / "bad.py"
    bad.write_text(source)
    assert _violations(bad)
