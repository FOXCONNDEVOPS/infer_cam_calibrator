#!/bin/bash
# filepath: /opt/infer_cam_calibrator/bin/install.sh
# Installation script for the camera calibration service on Ubuntu
# Sets up a Python 3.12 virtual environment, installs requirements, and configures as a systemd service

set -e  # Exit on error

# Ensure script is run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root" >&2
    exit 1
fi

# Define paths
APP_DIR="/opt/infer_cam_calibrator"
VENV_DIR="$APP_DIR/.venv"
LOG_DIR="/opt/kiosk_fw/logs"
SERVICE_NAME="calibration-service"
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"
PYTHON_VERSION="3.12.3"
PYENV_USER="$HOME/.pyenv"


# Create log directory if it doesn't exist
echo "Creating log directory..."
mkdir -p $LOG_DIR
chmod 755 $LOG_DIR



# Create virtual environment using pyenv's Python
echo "Setting up Python virtual environment..."
cd $APP_DIR
pyenv local $PYTHON_VERSION
PYENV_PYTHON="$PYENV_USER/versions/$PYTHON_VERSION/bin/python"
$PYENV_PYTHON -m venv $VENV_DIR

# Install requirements
echo "Installing Python dependencies..."
$VENV_DIR/bin/pip install --upgrade pip
$VENV_DIR/bin/pip install -r $APP_DIR/requirements.txt

# Create systemd service file
echo "Creating systemd service..."
cat > $SERVICE_FILE << EOF
[Unit]
Description=Camera Calibration Service
After=network.target mosquitto.service
Wants=mosquitto.service

[Service]
ExecStart=$VENV_DIR/bin/python $APP_DIR/calibration_service.py
WorkingDirectory=$APP_DIR
Restart=always
RestartSec=10
User=root
Group=root
Environment=PATH=$VENV_DIR/bin:$PYENV_USER/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYENV_ROOT=$PYENV_USER
StandardOutput=append:$LOG_DIR/calibration-service-output.log
StandardError=append:$LOG_DIR/calibration-service-error.log

[Install]
WantedBy=multi-user.target
EOF

# Set permissions and enable service
chmod 644 $SERVICE_FILE
systemctl daemon-reload
systemctl enable $SERVICE_NAME
systemctl start $SERVICE_NAME

echo "Installation complete! Camera calibration service has been installed and started."
echo "Python version: $($VENV_DIR/bin/python --version)"
echo "Service status: $(systemctl is-active $SERVICE_NAME)"
echo "Check logs at $LOG_DIR/calibration-service-output.log and $LOG_DIR/calibration-service-error.log"
