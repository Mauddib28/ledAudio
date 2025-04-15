#!/bin/bash

# Audio LED Visualization Environment Setup Script
echo "Setting up Python virtual environment for Audio LED Visualization..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create the virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please install python3-venv:"
        echo "sudo pacman -S python-virtualenv"
        exit 1
    fi
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required dependencies
echo "Installing required dependencies..."
pip install numpy pyyaml pyaudio matplotlib pygame

# Install optional dependencies based on hardware
echo "Do you want to install Raspberry Pi LED libraries? (y/n)"
read -r answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo "Installing Raspberry Pi LED libraries..."
    pip install adafruit-circuitpython-neopixel rpi_ws281x
fi

# Inform about activation
echo ""
echo "==============================================================="
echo "Environment setup complete!"
echo ""
echo "To activate the environment:"
echo "    source venv/bin/activate"
echo ""
echo "To deactivate the environment:"
echo "    deactivate"
echo ""
echo "Or use the provided scripts:"
echo "    ./activate_env.sh    # To activate"
echo "    ./deactivate_env.sh  # To deactivate"
echo "==============================================================="

# Make all scripts executable
chmod +x activate_env.sh deactivate_env.sh

# Return to normal if we were in the virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi 