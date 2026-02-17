#!/bin/bash
# PrivacyFlow Installation Script for Raspberry Pi
# 
# Usage:
#   chmod +x scripts/install_rpi.sh
#   ./scripts/install_rpi.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  PrivacyFlow Installation Script${NC}"
echo -e "${GREEN}  Privacy-Preserving Person Detection${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Raspberry Pi${NC}"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "Python version: ${GREEN}$PYTHON_VERSION${NC}"

if [[ "$PYTHON_VERSION" < "3.9" ]]; then
    echo -e "${RED}Error: Python 3.9+ required${NC}"
    exit 1
fi

# Update system
echo -e "\n${YELLOW}Step 1: Updating system packages...${NC}"
sudo apt-get update

# Install system dependencies
echo -e "\n${YELLOW}Step 2: Installing system dependencies...${NC}"
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libhdf5-dev \
    v4l-utils

# Add user to video group for camera access
echo -e "\n${YELLOW}Step 3: Configuring camera access...${NC}"
sudo usermod -aG video $USER

# Create virtual environment
echo -e "\n${YELLOW}Step 4: Creating Python virtual environment...${NC}"
INSTALL_DIR=$(pwd)

if [ -d "venv" ]; then
    echo "Virtual environment already exists"
else
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Step 5: Upgrading pip...${NC}"
pip install --upgrade pip wheel setuptools

# Install Python dependencies
echo -e "\n${YELLOW}Step 6: Installing Python dependencies...${NC}"
pip install -r requirements.txt

# For Raspberry Pi, install optimized numpy/opencv if available
if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo -e "\n${YELLOW}Installing Raspberry Pi optimized packages...${NC}"
    pip install picamera2 2>/dev/null || true
fi

# Download model placeholders
echo -e "\n${YELLOW}Step 7: Setting up models...${NC}"
python scripts/download_models.py --lightweight

# Create necessary directories
echo -e "\n${YELLOW}Step 8: Creating directories...${NC}"
mkdir -p data/exports
mkdir -p logs
mkdir -p models

# Set up systemd service
echo -e "\n${YELLOW}Step 9: Setting up systemd service...${NC}"
read -p "Install as systemd service? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Update service file with current directory
    sed -i "s|/home/pi/privacyflow|$INSTALL_DIR|g" privacyflow.service
    sed -i "s|User=pi|User=$USER|g" privacyflow.service
    
    sudo cp privacyflow.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable privacyflow
    
    echo -e "${GREEN}Service installed!${NC}"
    echo "  Start: sudo systemctl start privacyflow"
    echo "  Stop:  sudo systemctl stop privacyflow"
    echo "  Logs:  sudo journalctl -u privacyflow -f"
fi

# Test camera
echo -e "\n${YELLOW}Step 10: Testing camera...${NC}"
if v4l2-ctl --list-devices 2>/dev/null; then
    echo -e "${GREEN}Cameras found:${NC}"
    v4l2-ctl --list-devices
else
    echo -e "${YELLOW}No cameras detected. Make sure camera is connected.${NC}"
fi

# Run tests
echo -e "\n${YELLOW}Step 11: Running tests...${NC}"
python -m pytest tests/test_privacy.py -v --tb=short || true

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "To run PrivacyFlow:"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo -e "  ${YELLOW}python main.py --station-id station_001 --camera 0${NC}"
echo ""
echo -e "Or with API enabled:"
echo -e "  ${YELLOW}python main.py --station-id station_001 --camera 0 --api-port 5000${NC}"
echo ""
echo -e "${RED}Important:${NC} You may need to log out and back in for camera group access."
echo ""
echo -e "For real detection, you need to:"
echo -e "  1. Download/train actual ONNX models"
echo -e "  2. Place them in the models/ directory"
echo -e "  3. Update detector.py with real model inference"
