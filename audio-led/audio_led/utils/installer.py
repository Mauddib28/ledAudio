#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####
# Created on:        2025-04-15
# Created by:        AI Assistant
# Last modified on:  2025-04-15
####

#--------------------------------------
#   MODULE DESCRIPTION
#--------------------------------------

# This module handles the installation of dependencies for the Audio LED Visualization System.
# It checks for required dependencies based on the environment and installs missing packages,
# making the system easier to set up on different platforms.

#--------------------------------------
#       IMPORTS
#--------------------------------------

import os
import sys
import logging
import subprocess
import importlib
import platform
import shutil
from pathlib import Path

# Local imports for system type constants
try:
    from audio_led.common.environment import (
        SYSTEM_RASPBERRY_PI, SYSTEM_PICO_W,
        SYSTEM_REALTEK_BW16, SYSTEM_UNIX
    )
except ImportError:
    # Fallback constants if module not available
    SYSTEM_RASPBERRY_PI = "raspberry_pi"
    SYSTEM_PICO_W = "pico_w"
    SYSTEM_REALTEK_BW16 = "realtek_bw16"
    SYSTEM_UNIX = "unix"

#--------------------------------------
#       CONSTANTS
#--------------------------------------

# Required packages for different system types
REQUIRED_PACKAGES = {
    SYSTEM_RASPBERRY_PI: {
        'python': ['numpy', 'pyaudio', 'pigpio', 'pyyaml', 'wave', 'gpiozero'],
        'system': ['python3-dev', 'libasound2-dev', 'portaudio19-dev', 'pigpio']
    },
    SYSTEM_UNIX: {
        'python': ['numpy', 'pyaudio', 'pyyaml', 'wave', 'pygame'],
        'system': ['python3-dev', 'libasound2-dev', 'portaudio19-dev']
    },
    # Pico W uses different installation methods (not pip or apt)
    SYSTEM_PICO_W: {
        'micropython': ['machine', 'uos', 'urandom', 'time']
    },
    # RealTek BW-16 package requirements (placeholder)
    SYSTEM_REALTEK_BW16: {
        'python': ['numpy', 'pyaudio', 'pyyaml'],
        'system': []
    }
}

# Optional packages for extra features
OPTIONAL_PACKAGES = {
    'bluetooth': {
        'python': ['bluepy', 'pybluez'],
        'system': ['libbluetooth-dev', 'bluetooth']
    },
    'display': {
        'python': ['pygame', 'pillow'],
        'system': []
    },
    'file_formats': {
        'python': ['pydub'],
        'system': ['ffmpeg']
    }
}

#--------------------------------------
#       FUNCTIONS
#--------------------------------------

def is_package_installed(package_name):
    """
    Check if a Python package is installed.
    
    Args:
        package_name (str): Name of the package to check
        
    Returns:
        bool: True if the package is installed, False otherwise
    """
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def check_system_dependency(package_name):
    """
    Check if a system package is installed.
    
    Args:
        package_name (str): Name of the system package to check
        
    Returns:
        bool: True if the package is installed, False otherwise
    """
    # Different package managers for different platforms
    if platform.system() == 'Linux':
        if os.path.exists('/usr/bin/apt'):
            # Debian/Ubuntu
            try:
                result = subprocess.run(
                    ['dpkg', '-s', package_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False
                )
                return result.returncode == 0
            except (subprocess.SubprocessError, FileNotFoundError):
                return False
        elif os.path.exists('/usr/bin/yum'):
            # RedHat/CentOS
            try:
                result = subprocess.run(
                    ['rpm', '-q', package_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False
                )
                return result.returncode == 0
            except (subprocess.SubprocessError, FileNotFoundError):
                return False
        elif os.path.exists('/usr/bin/pacman'):
            # Arch Linux
            try:
                result = subprocess.run(
                    ['pacman', '-Q', package_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False
                )
                return result.returncode == 0
            except (subprocess.SubprocessError, FileNotFoundError):
                return False
    
    # For other systems, assume it's installed (harder to check)
    # Or we could return False to be safe
    return True

def install_package_pip(package_name):
    """
    Install a Python package using pip.
    
    Args:
        package_name (str): Name of the package to install
        
    Returns:
        bool: True if installation was successful, False otherwise
    """
    try:
        logging.info(f"Installing Python package: {package_name}")
        
        # Use subprocess to run pip
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--user', package_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        
        if result.returncode == 0:
            logging.info(f"Successfully installed {package_name}")
            return True
        else:
            error = result.stderr.decode('utf-8')
            logging.error(f"Failed to install {package_name}: {error}")
            return False
            
    except Exception as e:
        logging.error(f"Error installing {package_name}: {e}")
        return False

def install_system_package(package_name):
    """
    Install a system package using the appropriate package manager.
    
    Args:
        package_name (str): Name of the system package to install
        
    Returns:
        bool: True if installation was successful, False otherwise
    """
    try:
        if platform.system() == 'Linux':
            if os.path.exists('/usr/bin/apt'):
                # Debian/Ubuntu
                cmd = ['sudo', 'apt-get', 'install', '-y', package_name]
            elif os.path.exists('/usr/bin/yum'):
                # RedHat/CentOS
                cmd = ['sudo', 'yum', 'install', '-y', package_name]
            elif os.path.exists('/usr/bin/pacman'):
                # Arch Linux
                cmd = ['sudo', 'pacman', '-S', '--noconfirm', package_name]
            else:
                logging.error("Unsupported package manager")
                return False
            
            logging.info(f"Installing system package: {package_name}")
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            
            if result.returncode == 0:
                logging.info(f"Successfully installed {package_name}")
                return True
            else:
                error = result.stderr.decode('utf-8')
                logging.error(f"Failed to install {package_name}: {error}")
                return False
        else:
            logging.error(f"System package installation not supported on {platform.system()}")
            return False
            
    except Exception as e:
        logging.error(f"Error installing {package_name}: {e}")
        return False

def check_and_install_dependencies(system_type):
    """
    Check for required dependencies and install them if missing.
    
    Args:
        system_type (str): The detected system type ('linux', 'darwin', 'windows', etc.)
        
    Returns:
        bool: True if all dependencies are installed or were successfully installed, False otherwise
    """
    logger.info(f"Checking dependencies for system type: {system_type}")
    
    # Check if we're running on a supported system
    if system_type.lower() not in ['linux', 'darwin', 'windows']:
        logger.warning(f"Unsupported system type: {system_type}")
        return False
    
    success = True
    
    # Check for Python dependencies
    python_deps = [
        'numpy',
        'scipy',
        'pygame',
        'pyaudio',
        'pydub',
        'sounddevice',
        'librosa'
    ]
    
    for dep in python_deps:
        if not check_python_dependency(dep):
            if not install_python_dependency(dep):
                logger.warning(f"Failed to install {dep}")
                success = False
    
    # Check for system dependencies
    if system_type.lower() == 'linux':
        # Check for distribution
        distro = get_linux_distro()
        logger.info(f"Detected Linux distribution: {distro}")
        
        # Check for required system packages based on distribution
        if distro == 'debian' or distro == 'ubuntu':
            system_deps = [
                'python3-dev',
                'python3-numpy',
                'python3-scipy',
                'libasound2-dev',
                'portaudio19-dev',
                'libportaudio2',
                'ffmpeg'  # Add ffmpeg for audio playback
            ]
            
            for dep in system_deps:
                if not check_system_dependency_debian(dep):
                    if not install_system_dependency_debian(dep):
                        logger.warning(f"Failed to install {dep}")
                        success = False
                        
        elif distro == 'arch' or distro == 'manjaro':
            system_deps = [
                'python-numpy',
                'python-scipy',
                'portaudio',
                'ffmpeg'  # Add ffmpeg for audio playback
            ]
            
            for dep in system_deps:
                if not check_system_dependency_arch(dep):
                    if not install_system_dependency_arch(dep):
                        logger.warning(f"Failed to install {dep}")
                        success = False
        else:
            # For other Linux distributions, just check for Python dependencies
            logger.warning(f"Unsupported Linux distribution: {distro}")
            logger.warning("Only checking Python dependencies")
            
    elif system_type.lower() == 'darwin':
        # Check for Homebrew
        if not check_system_dependency_mac('brew'):
            logger.warning("Homebrew not found, cannot check system dependencies")
            logger.warning("Visit https://brew.sh/ to install Homebrew")
            success = False
        else:
            # Check for required system packages
            system_deps = [
                'portaudio',
                'ffmpeg'  # Add ffmpeg for audio playback
            ]
            
            for dep in system_deps:
                if not check_system_dependency_mac(dep):
                    if not install_system_dependency_mac(dep):
                        logger.warning(f"Failed to install {dep}")
                        success = False
    
    # For Windows, we rely mostly on Python packages
    # Windows-specific checks would go here
    
    return success

def get_missing_dependencies(system_type, features=None):
    """
    Get a list of missing dependencies without installing them.
    
    Args:
        system_type (str): Type of system (SYSTEM_RASPBERRY_PI, SYSTEM_UNIX, etc.)
        features (list, optional): List of optional features to include
        
    Returns:
        dict: Dictionary of missing dependencies by category
    """
    if system_type not in REQUIRED_PACKAGES:
        logging.error(f"Unsupported system type: {system_type}")
        return {}
    
    missing = {
        'python': [],
        'system': []
    }
    
    # Get required packages for the system type
    required = REQUIRED_PACKAGES.get(system_type, {})
    
    # Add optional packages based on requested features
    if features:
        for feature in features:
            if feature in OPTIONAL_PACKAGES:
                for pkg_type, packages in OPTIONAL_PACKAGES[feature].items():
                    if pkg_type in required:
                        required[pkg_type].extend(packages)
                    else:
                        required[pkg_type] = packages
    
    # Check Python packages
    if 'python' in required:
        for package in required['python']:
            if not is_package_installed(package):
                missing['python'].append(package)
    
    # Check system packages
    if 'system' in required:
        for package in required['system']:
            if not check_system_dependency(package):
                missing['system'].append(package)
    
    return missing

def print_installation_instructions(missing_deps):
    """
    Print instructions for manual installation of missing dependencies.
    
    Args:
        missing_deps (dict): Dictionary of missing dependencies by category
    """
    if not missing_deps.get('python') and not missing_deps.get('system'):
        print("All dependencies are installed!")
        return
    
    print("\nThe following dependencies need to be installed:")
    
    if missing_deps.get('python'):
        print("\nPython packages:")
        print(f"pip install {' '.join(missing_deps['python'])}")
    
    if missing_deps.get('system'):
        print("\nSystem packages:")
        if os.path.exists('/usr/bin/apt'):
            print(f"sudo apt-get install {' '.join(missing_deps['system'])}")
        elif os.path.exists('/usr/bin/yum'):
            print(f"sudo yum install {' '.join(missing_deps['system'])}")
        elif os.path.exists('/usr/bin/pacman'):
            print(f"sudo pacman -S {' '.join(missing_deps['system'])}")
        else:
            print(f"Please install these packages with your system's package manager: {', '.join(missing_deps['system'])}")

def get_linux_distro():
    """
    Detect the Linux distribution.
    
    Returns:
        str: Name of the Linux distribution in lowercase, or 'unknown'
    """
    try:
        # Try to use /etc/os-release first (most modern distros)
        if os.path.exists('/etc/os-release'):
            with open('/etc/os-release', 'r') as f:
                for line in f:
                    if line.startswith('ID='):
                        return line.split('=')[1].strip().strip('"').lower()
        
        # Otherwise, try to use the platform module
        if hasattr(platform, 'linux_distribution'):
            distro = platform.linux_distribution()[0].lower()
            if 'debian' in distro:
                return 'debian'
            elif 'ubuntu' in distro:
                return 'ubuntu'
            elif 'arch' in distro:
                return 'arch'
            elif 'manjaro' in distro:
                return 'manjaro'
            else:
                return distro
                
        # Fallback to check for common distro-specific files
        if os.path.exists('/etc/debian_version'):
            return 'debian'
        elif os.path.exists('/etc/arch-release'):
            return 'arch'
            
    except Exception as e:
        logger.error(f"Error detecting Linux distribution: {e}")
    
    return 'unknown'

def check_python_dependency(package_name):
    """
    Check if a Python package is installed.
    
    Args:
        package_name (str): Name of the package to check
        
    Returns:
        bool: True if the package is installed, False otherwise
    """
    try:
        importlib.import_module(package_name)
        logger.info(f"Python package {package_name} is installed")
        return True
    except ImportError:
        logger.info(f"Python package {package_name} is not installed")
        return False

def install_python_dependency(package_name):
    """
    Install a Python package using pip.
    
    Args:
        package_name (str): Name of the package to install
        
    Returns:
        bool: True if installation succeeded, False otherwise
    """
    try:
        logger.info(f"Installing Python package: {package_name}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        logger.info(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing {package_name}: {e}")
        return False

def check_system_dependency_debian(package_name):
    """
    Check if a system package is installed on Debian/Ubuntu.
    
    Args:
        package_name (str): Name of the package to check
        
    Returns:
        bool: True if the package is installed, False otherwise
    """
    try:
        result = subprocess.run(['dpkg', '-s', package_name], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        if result.returncode == 0:
            logger.info(f"System package {package_name} is installed")
            return True
        else:
            logger.info(f"System package {package_name} is not installed")
            return False
    except Exception as e:
        logger.error(f"Error checking for system package {package_name}: {e}")
        return False

def install_system_dependency_debian(package_name):
    """
    Install a system package on Debian/Ubuntu.
    
    Args:
        package_name (str): Name of the package to install
        
    Returns:
        bool: True if installation succeeded, False otherwise
    """
    try:
        logger.info(f"Installing system package: {package_name}")
        
        # Check if we have sudo privileges
        if os.geteuid() == 0:
            # We're running as root
            subprocess.check_call(['apt-get', 'update'])
            subprocess.check_call(['apt-get', 'install', '-y', package_name])
        else:
            # We need sudo
            if shutil.which('sudo'):
                subprocess.check_call(['sudo', 'apt-get', 'update'])
                subprocess.check_call(['sudo', 'apt-get', 'install', '-y', package_name])
            else:
                logger.error("Cannot install system packages without root privileges")
                return False
                
        logger.info(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing {package_name}: {e}")
        return False

def check_system_dependency_arch(package_name):
    """
    Check if a system package is installed on Arch Linux.
    
    Args:
        package_name (str): Name of the package to check
        
    Returns:
        bool: True if the package is installed, False otherwise
    """
    try:
        result = subprocess.run(['pacman', '-Q', package_name], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        if result.returncode == 0:
            logger.info(f"System package {package_name} is installed")
            return True
        else:
            logger.info(f"System package {package_name} is not installed")
            return False
    except Exception as e:
        logger.error(f"Error checking for system package {package_name}: {e}")
        return False

def install_system_dependency_arch(package_name):
    """
    Install a system package on Arch Linux.
    
    Args:
        package_name (str): Name of the package to install
        
    Returns:
        bool: True if installation succeeded, False otherwise
    """
    try:
        logger.info(f"Installing system package: {package_name}")
        
        # Check if we have sudo privileges
        if os.geteuid() == 0:
            # We're running as root
            subprocess.check_call(['pacman', '-Sy', '--noconfirm', package_name])
        else:
            # We need sudo
            if shutil.which('sudo'):
                subprocess.check_call(['sudo', 'pacman', '-Sy', '--noconfirm', package_name])
            else:
                logger.error("Cannot install system packages without root privileges")
                return False
                
        logger.info(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing {package_name}: {e}")
        return False

def check_system_dependency_mac(package_name):
    """
    Check if a system package is installed on macOS using Homebrew.
    
    Args:
        package_name (str): Name of the package to check
        
    Returns:
        bool: True if the package is installed, False otherwise
    """
    # Special case for Homebrew itself
    if package_name == 'brew':
        return shutil.which('brew') is not None
        
    try:
        result = subprocess.run(['brew', 'list', package_name], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        if result.returncode == 0:
            logger.info(f"System package {package_name} is installed")
            return True
        else:
            logger.info(f"System package {package_name} is not installed")
            return False
    except Exception as e:
        logger.error(f"Error checking for system package {package_name}: {e}")
        return False

def install_system_dependency_mac(package_name):
    """
    Install a system package on macOS using Homebrew.
    
    Args:
        package_name (str): Name of the package to install
        
    Returns:
        bool: True if installation succeeded, False otherwise
    """
    try:
        logger.info(f"Installing system package: {package_name}")
        subprocess.check_call(['brew', 'install', package_name])
        logger.info(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing {package_name}: {e}")
        return False

# Run checks and show instructions if this script is run directly
if __name__ == "__main__":
    from audio_led.common.environment import detect_environment
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Detect environment
    env_info = detect_environment()
    system_type = env_info['system_type']
    
    # Get missing dependencies
    missing = get_missing_dependencies(system_type)
    
    # Print installation instructions
    print(f"Detected system type: {system_type}")
    print_installation_instructions(missing)
    
    # Install dependencies if requested
    if '--install' in sys.argv:
        print("\nInstalling missing dependencies...")
        if check_and_install_dependencies(system_type):
            print("All dependencies installed successfully!")
        else:
            print("Some dependencies could not be installed.") 