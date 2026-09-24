#!/bin/bash
# Installation script for the camera calibration service on Ubuntu.
# Called by kiosk_fw as `sudo bin/install.sh` (no arguments) on a fresh clone at
# /opt/infer_cam_calibrator. This interface is stable: do not add required arguments.
#
# Bootstraps the uv release pinned in pyproject.toml ([tool.uv]) into .tools/uv,
# verified by sha256, syncs .venv from uv.lock, and installs the systemd service.
# Never uses a uv found on PATH or kiosk_fw's uv.
#
# Testing hook: INSTALL_SH_UV_ONLY_DIR=<dir> bin/install.sh only bootstraps uv into
# <dir>/.tools/uv and exits (no root needed, nothing else touched).

set -e  # Exit on error

# Define paths
APP_DIR="/opt/infer_cam_calibrator"
LOG_DIR="/opt/kiosk_fw/logs"
SERVICE_NAME="calibration-service"
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYPROJECT="$REPO_DIR/pyproject.toml"

# The uv pin lives in pyproject.toml only: `required-version = "==X.Y.Z"` and the
# `# sha256 <hex>` comment in [tool.uv].
UV_VERSION="$(sed -n 's/^required-version = "==\([0-9][0-9.]*\)"$/\1/p' "$PYPROJECT")"
UV_SHA256="$(sed -n 's/^#[[:space:]]*sha256[[:space:]]\{1,\}\([0-9a-f]\{64\}\)[[:space:]]*$/\1/p' "$PYPROJECT")"
UV_TARGET="x86_64-unknown-linux-gnu"
UV_URL="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$UV_TARGET.tar.gz"

if [ -z "$UV_VERSION" ] || [ -z "$UV_SHA256" ]; then
    echo "Could not read the uv version and sha256 from $PYPROJECT" >&2
    exit 1
fi

# Install the pinned uv into $1/.tools/uv unless that exact version is already there.
bootstrap_uv() {
    local tools_dir="$1/.tools"
    local uv_bin="$tools_dir/uv"

    if [ -x "$uv_bin" ] && [ "$("$uv_bin" --version 2>/dev/null | awk '{print $2}')" = "$UV_VERSION" ]; then
        echo "uv $UV_VERSION already installed at $uv_bin"
        return 0
    fi

    if [ "$(uname -m)" != "x86_64" ]; then
        echo "Only x86_64 is supported (pinned uv asset is $UV_TARGET), got $(uname -m)" >&2
        exit 1
    fi

    echo "Installing uv $UV_VERSION into $tools_dir..."
    local tmp
    tmp="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" EXIT

    curl -fsSL --retry 3 -o "$tmp/uv.tar.gz" "$UV_URL"
    echo "$UV_SHA256  $tmp/uv.tar.gz" | sha256sum -c --quiet -
    tar -xzf "$tmp/uv.tar.gz" -C "$tmp"

    mkdir -p "$tools_dir"
    install -m 755 "$tmp/uv-$UV_TARGET/uv" "$uv_bin"
    rm -rf "$tmp"
    trap - EXIT

    echo "Installed $("$uv_bin" --version)"
}

if [ -n "$INSTALL_SH_UV_ONLY_DIR" ]; then
    bootstrap_uv "$INSTALL_SH_UV_ONLY_DIR"
    exit 0
fi

# Ensure script is run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root" >&2
    exit 1
fi

VENV_DIR="$APP_DIR/.venv"
UV="$APP_DIR/.tools/uv"

# Create log directory if it doesn't exist
echo "Creating log directory..."
mkdir -p "$LOG_DIR"
chmod 755 "$LOG_DIR"

bootstrap_uv "$APP_DIR"

# uv installs the pinned Python (.python-version) in root's default uv location
# and builds .venv exactly from uv.lock.
echo "Installing Python and dependencies with uv..."
cd "$APP_DIR"
unset VIRTUAL_ENV
"$UV" sync --frozen --no-dev --managed-python

# Create systemd service file
echo "Creating systemd service..."
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Camera Calibration Service
After=network.target mosquitto.service
Wants=mosquitto.service

[Service]
ExecStart=$VENV_DIR/bin/python -m infer_cam_calibrator.calibration_service
WorkingDirectory=$APP_DIR
Restart=always
RestartSec=10
User=root
Group=root
Environment=PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin
StandardOutput=append:$LOG_DIR/calibration-service-output.log
StandardError=append:$LOG_DIR/calibration-service-error.log

[Install]
WantedBy=multi-user.target
EOF

# Set permissions and enable service
chmod 644 "$SERVICE_FILE"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo "Installation complete! Camera calibration service has been installed and started."
echo "Python version: $("$VENV_DIR/bin/python" --version)"
echo "Service status: $(systemctl is-active "$SERVICE_NAME")"
echo "Check logs at $LOG_DIR/calibration-service-output.log and $LOG_DIR/calibration-service-error.log"
