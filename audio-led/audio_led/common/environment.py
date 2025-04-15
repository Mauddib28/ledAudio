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

# This module handles environment detection for the Audio LED Visualization System.
# It identifies the hardware platform (Raspberry Pi, Pico W, RealTek BW-16, or generic Unix)
# and checks for the availability of various hardware features and software dependencies.

#--------------------------------------
#       IMPORTS
#--------------------------------------

import os
import sys
import platform
import logging
import subprocess
import importlib
from pathlib import Path

#--------------------------------------
#       CONSTANTS
#--------------------------------------

# System types
SYSTEM_RASPBERRY_PI = "raspberry_pi"
SYSTEM_PICO_W = "pico_w"
SYSTEM_REALTEK_BW16 = "realtek_bw16"
SYSTEM_UNIX = "unix"
SYSTEM_UNKNOWN = "unknown"

# Hardware capabilities
CAPABILITY_GPIO = "gpio"
CAPABILITY_I2S = "i2s"
CAPABILITY_PWM = "pwm"
CAPABILITY_AUDIO_INPUT = "audio_input"
CAPABILITY_AUDIO_OUTPUT = "audio_output"
CAPABILITY_DISPLAY = "display"
CAPABILITY_BLUETOOTH = "bluetooth"

# Required libraries for different features
LIBRARIES = {
    "audio_processing": ["numpy", "wave", "pyaudio"],
    "led_control": ["pigpio", "spidev", "gpiozero"],
    "gui": ["pygame", "tkinter"],
    "file_formats": ["wave", "pydub"],
    "bluetooth": ["bluepy", "pybluez"]
}

#--------------------------------------
#       FUNCTIONS
#--------------------------------------

def is_module_available(module_name):
    """
    Check if a Python module is available.
    
    Args:
        module_name (str): Name of the module to check
        
    Returns:
        bool: True if the module is available, False otherwise
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def check_command_availability(command):
    """
    Check if a command is available in the system PATH.
    
    Args:
        command (str): Command to check
        
    Returns:
        bool: True if the command is available, False otherwise
    """
    try:
        subprocess.run(
            ["which", command], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=False
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
    
def detect_raspberry_pi():
    """
    Detect if running on a Raspberry Pi.
    
    Returns:
        bool: True if running on a Raspberry Pi, False otherwise
    """
    # Check for Raspberry Pi specific files
    if os.path.exists('/proc/device-tree/model'):
        with open('/proc/device-tree/model') as f:
            model = f.read()
            if 'Raspberry Pi' in model:
                return True
    
    # Alternative detection methods
    try:
        output = subprocess.check_output(['cat', '/proc/cpuinfo']).decode('utf-8')
        if 'BCM' in output and 'Raspberry Pi' in output:
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
        
    return False

def detect_pico_w():
    """
    Detect if running on a Raspberry Pi Pico W.
    
    Returns:
        bool: True if running on a Pico W, False otherwise
    """
    # Check if running MicroPython
    if hasattr(sys, 'implementation') and sys.implementation.name == 'micropython':
        try:
            # Try to import Pico W specific library
            import machine
            # Check model name or other Pico W specific identifiers
            if hasattr(machine, 'unique_id') and "Pico W" in str(os.uname()):
                return True
        except ImportError:
            pass
    return False

def detect_realtek_bw16():
    """
    Detect if running on a RealTek BW-16.
    
    Returns:
        bool: True if running on a RealTek BW-16, False otherwise
    """
    # Implementation will depend on specific identifiers for RealTek BW-16
    # This is a placeholder
    try:
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model') as f:
                model = f.read()
                if 'RealTek BW-16' in model:
                    return True
    except (IOError, OSError):
        pass
    
    return False

def detect_gpio_capability():
    """
    Detect if GPIO pins are available.
    
    Returns:
        bool: True if GPIO is available, False otherwise
    """
    # Check for GPIO libraries or hardware access
    if detect_raspberry_pi():
        return (os.path.exists('/sys/class/gpio') or 
                is_module_available('RPi.GPIO') or 
                is_module_available('gpiozero'))
    elif detect_pico_w():
        return is_module_available('machine')
    return False

def detect_audio_capability():
    """
    Detect audio input/output capabilities.
    
    Returns:
        dict: Dictionary with audio_input and audio_output booleans
    """
    audio_input = False
    audio_output = False
    
    # Check for ALSA or other audio systems
    if os.path.exists('/proc/asound'):
        audio_output = True
        # Check for input devices (microphones, line-in)
        try:
            if is_module_available('pyaudio'):
                import pyaudio
                p = pyaudio.PyAudio()
                input_device_count = 0
                for i in range(p.get_device_count()):
                    device_info = p.get_device_info_by_index(i)
                    if device_info.get('maxInputChannels') > 0:
                        input_device_count += 1
                p.terminate()
                audio_input = input_device_count > 0
        except Exception as e:
            logging.debug(f"Error detecting audio devices: {e}")
    
    # Check for alternative audio libraries
    if is_module_available('sounddevice') or is_module_available('alsaaudio'):
        audio_output = True
        audio_input = True
    
    return {
        CAPABILITY_AUDIO_INPUT: audio_input,
        CAPABILITY_AUDIO_OUTPUT: audio_output
    }

def detect_display_capability():
    """
    Detect if a display is available for visualization.
    
    Returns:
        bool: True if a display is available, False otherwise
    """
    # Check for display capabilities (X server, SDL, etc.)
    if 'DISPLAY' in os.environ:
        return True
    
    # On Raspberry Pi, check for SPI (often used for displays)
    if detect_raspberry_pi() and os.path.exists('/dev/spidev0.0'):
        return True
    
    # Check for various display libraries
    for lib in ['pygame', 'tkinter', 'PyQt5', 'kivy']:
        if is_module_available(lib):
            return True
            
    return False

def detect_bluetooth_capability():
    """
    Detect if Bluetooth capabilities are available.
    
    Returns:
        bool: True if Bluetooth is available, False otherwise
    """
    # Check for Bluetooth hardware and software
    if os.path.exists('/sys/class/bluetooth'):
        return True
        
    # Check for Bluetooth libraries
    for lib in ['bluetooth', 'bluepy', 'pybluez']:
        if is_module_available(lib):
            return True
            
    # Try hciconfig command
    try:
        result = subprocess.run(
            ['hciconfig'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=False
        )
        if result.returncode == 0 and b'hci' in result.stdout:
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
            
    return False

def check_installed_libraries():
    """
    Check which required libraries are installed.
    
    Returns:
        dict: Dictionary mapping feature categories to available library sets
    """
    available_libraries = {}
    
    for category, libs in LIBRARIES.items():
        available_libraries[category] = set()
        for lib in libs:
            if is_module_available(lib):
                available_libraries[category].add(lib)
    
    return available_libraries

def detect_environment():
    """
    Detect the environment and return information about it.
    
    Returns:
        dict: Environment information including system type and capabilities
    """
    # Detect system type
    system_type = SYSTEM_UNKNOWN
    if detect_pico_w():
        system_type = SYSTEM_PICO_W
    elif detect_raspberry_pi():
        system_type = SYSTEM_RASPBERRY_PI
    elif detect_realtek_bw16():
        system_type = SYSTEM_REALTEK_BW16
    elif platform.system() in ['Linux', 'Darwin', 'FreeBSD']:
        system_type = SYSTEM_UNIX
    
    # Detect system capabilities
    capabilities = {
        CAPABILITY_GPIO: detect_gpio_capability(),
        CAPABILITY_PWM: detect_gpio_capability(),  # Simplified assumption
        CAPABILITY_DISPLAY: detect_display_capability(),
        CAPABILITY_BLUETOOTH: detect_bluetooth_capability()
    }
    
    # Add audio capabilities
    audio_capabilities = detect_audio_capability()
    capabilities.update(audio_capabilities)
    
    # Check installed libraries
    available_libraries = check_installed_libraries()
    
    # Check if all required dependencies are installed
    dependencies_installed = True
    if system_type == SYSTEM_UNIX:
        # For Unix, we need audio processing, and either LED control or GUI
        required_categories = ['audio_processing', 'file_formats']
        if capabilities[CAPABILITY_DISPLAY]:
            required_categories.append('gui')
        if capabilities[CAPABILITY_GPIO]:
            required_categories.append('led_control')
            
        for category in required_categories:
            if not available_libraries.get(category, set()):
                dependencies_installed = False
                break
    
    return {
        'system_type': system_type,
        'platform_info': {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'python': platform.python_version()
        },
        'capabilities': capabilities,
        'available_libraries': available_libraries,
        'dependencies_installed': dependencies_installed
    }

# If run directly, print environment information
if __name__ == "__main__":
    env_info = detect_environment()
    print(f"System type: {env_info['system_type']}")
    print(f"Platform: {env_info['platform_info']}")
    print("\nCapabilities:")
    for cap, available in env_info['capabilities'].items():
        print(f"  {cap}: {available}")
    print("\nAvailable libraries:")
    for category, libs in env_info['available_libraries'].items():
        print(f"  {category}: {', '.join(libs) if libs else 'None'}") 