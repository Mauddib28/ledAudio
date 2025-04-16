#!/bin/bash

# Audio-LED Dependencies Installer
# This script attempts to install all required dependencies for the Audio-LED project
# It supports multiple Linux distributions including Ubuntu, Debian, and Arch Linux

echo "Audio-LED Dependencies Installer"
echo "================================"
echo

# Detect distribution
DISTRO=""
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
elif command -v lsb_release >/dev/null 2>&1; then
    DISTRO=$(lsb_release -si | tr '[:upper:]' '[:lower:]')
else
    # Try to detect by checking for package managers
    if command -v apt-get >/dev/null 2>&1; then
        DISTRO="debian"
    elif command -v pacman >/dev/null 2>&1; then
        DISTRO="arch"
    fi
fi

echo "Detected distribution: $DISTRO"

# Function to install dependencies on Debian/Ubuntu
install_debian() {
    echo "Installing dependencies for Debian/Ubuntu..."
    
    # Update package lists
    sudo apt-get update
    
    # Install essential dependencies
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-numpy \
        python3-pyaudio \
        ffmpeg \
        libsndfile1
    
    # Install optional Python packages
    pip3 install --user pydub soundfile scipy librosa
    
    echo "Dependencies installed successfully!"
}

# Function to install dependencies on Arch Linux
install_arch() {
    echo "Installing dependencies for Arch Linux..."
    
    # Update package lists
    sudo pacman -Sy
    
    # Install essential dependencies
    sudo pacman -S --noconfirm \
        python \
        python-pip \
        python-virtualenv \
        python-numpy \
        python-pyaudio \
        ffmpeg
    
    # Install optional Python packages
    pip install --user pydub soundfile scipy librosa
    
    echo "Dependencies installed successfully!"
}

# Function for generic installation when distribution cannot be determined
install_generic() {
    echo "Could not determine your distribution."
    echo "Attempting generic installation..."
    
    # Check if pip is available
    if command -v pip3 >/dev/null 2>&1; then
        pip3 install --user numpy pyaudio pydub soundfile scipy librosa
    elif command -v pip >/dev/null 2>&1; then
        pip install --user numpy pyaudio pydub soundfile scipy librosa
    else
        echo "Error: pip not found. Please install pip first."
        exit 1
    fi
    
    echo "Python packages installed. You may still need to install:"
    echo "- ffmpeg (for audio file processing)"
    echo "- Python development packages (for some modules to compile)"
}

# Install dependencies based on detected distribution
case $DISTRO in
    debian|ubuntu|raspbian)
        install_debian
        ;;
    arch|manjaro)
        install_arch
        ;;
    *)
        echo "Unsupported or undetected distribution."
        echo "You may need to install dependencies manually."
        
        # Ask if user wants to try generic installation
        read -p "Would you like to try a generic installation? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_generic
        fi
        ;;
esac

echo
echo "If running the application fails due to missing dependencies, you can:"
echo "1. Run with '--no-venv' option to use system Python"
echo "2. Run with '--install-deps' option to try automatic dependency installation"
echo "3. Check the logs for specific missing packages"
echo

echo "Installation process complete!"
exit 0 