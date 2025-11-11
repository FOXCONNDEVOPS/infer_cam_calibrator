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

# Install pyenv if not already installed
if [ ! -d "$HOME/.pyenv" ]; then
    echo "Installing pyenv..."
    curl https://pyenv.run | bash
else
    echo "pyenv already installed"
fi

# Set up pyenv environment
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Install Python 3.12.3 via pyenv if not already installed
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "Installing Python $PYTHON_VERSION via pyenv..."
    pyenv install $PYTHON_VERSION
else
    echo "Python $PYTHON_VERSION already installed via pyenv"
fi

# Create symlink in /usr/bin if it doesn't exist
if [ ! -L "/usr/bin/python3.12" ]; then
    echo "Creating symlink /usr/bin/python3.12..."
    ln -sf "$HOME/.pyenv/versions/$PYTHON_VERSION/bin/python3.12" /usr/bin/python3.12
fi

# Verify Python installation
if ! /usr/bin/python3.12 --version; then
    echo "Python 3.12 installation failed!" >&2
    exit 1
fi

echo "Python 3.12 installed successfully: $(/usr/bin/python3.12 --version)"

# Create log directory if it doesn't exist
echo "Creating log directory..."
mkdir -p $LOG_DIR
chmod 755 $LOG_DIR

# Create and activate virtual environment
echo "Setting up Python virtual environment..."
/usr/bin/python3.12 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Install requirements
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r $APP_DIR/requirements.txt

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
Environment=PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin
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
echo "Service status: $(systemctl is-active $SERVICE_NAME)"
echo "Check logs at $LOG_DIR/calibration-service-output.log and $LOG_DIR/calibration-service-error.log"