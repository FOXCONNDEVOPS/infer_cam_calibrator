"""Static invariants of bin/install.sh, which kiosk_fw runs as `sudo bin/install.sh`."""

import re
import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "bin" / "install.sh"
PYPROJECT = REPO_ROOT / "pyproject.toml"

SCRIPT = INSTALL_SH.read_text()


def test_bash_syntax():
    subprocess.run(["bash", "-n", str(INSTALL_SH)], check=True)


def test_executable_with_set_e():
    assert INSTALL_SH.stat().st_mode & 0o111
    assert SCRIPT.startswith("#!/bin/bash\n")
    assert re.search(r"^set -e\b", SCRIPT, re.M)


def test_no_pyenv_or_requirements():
    assert "pyenv" not in SCRIPT.lower()
    assert "requirements.txt" not in SCRIPT
    assert not (REPO_ROOT / "requirements.txt").exists()


def test_paths_and_service_name():
    assert 'APP_DIR="/opt/infer_cam_calibrator"' in SCRIPT
    assert 'LOG_DIR="/opt/kiosk_fw/logs"' in SCRIPT
    assert 'SERVICE_NAME="calibration-service"' in SCRIPT
    assert 'mkdir -p "$LOG_DIR"' in SCRIPT


def test_unit_file():
    assert "ExecStart=$VENV_DIR/bin/python -m infer_cam_calibrator.calibration_service\n" in SCRIPT
    assert "WorkingDirectory=$APP_DIR\n" in SCRIPT
    assert "User=root\n" in SCRIPT
    assert "StandardOutput=append:$LOG_DIR/calibration-service-output.log\n" in SCRIPT
    assert "StandardError=append:$LOG_DIR/calibration-service-error.log\n" in SCRIPT


def test_uses_only_bundled_uv():
    assert 'UV="$APP_DIR/.tools/uv"' in SCRIPT
    assert '"$UV" sync --frozen --no-dev' in SCRIPT
    # Every uv invocation goes through an absolute path variable, never bare `uv` from PATH.
    for line in SCRIPT.splitlines():
        code = line.split("#", 1)[0].strip()
        if code.startswith("echo "):
            continue
        assert not re.search(r"(^|[\s;&|(])uvx?\s", code), line


def test_uv_pin_read_from_pyproject():
    """install.sh reads the pin from pyproject; check its sed patterns find the real values."""
    text = PYPROJECT.read_text()
    required = tomllib.loads(text)["tool"]["uv"]["required-version"]
    assert re.fullmatch(r"==\d+\.\d+\.\d+", required)

    version = subprocess.run(
        ["sed", "-n", r's/^required-version = "==\([0-9][0-9.]*\)"$/\1/p', str(PYPROJECT)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    sha = subprocess.run(
        ["sed", "-n", r"s/^#[[:space:]]*sha256[[:space:]]\{1,\}\([0-9a-f]\{64\}\)[[:space:]]*$/\1/p", str(PYPROJECT)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    assert version == required.removeprefix("==")
    assert re.fullmatch(r"[0-9a-f]{64}", sha)
    # The same patterns must be the ones install.sh uses.
    assert r's/^required-version = "==\([0-9][0-9.]*\)"$/\1/p' in SCRIPT
    assert r"s/^#[[:space:]]*sha256[[:space:]]\{1,\}\([0-9a-f]\{64\}\)[[:space:]]*$/\1/p" in SCRIPT
