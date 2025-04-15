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

# This module provides device management functionality for the Audio LED Visualization System.
# It handles:
# - Detection of available hardware devices (audio input/output, LEDs)
# - Initialization and configuration of hardware devices
# - Abstraction layer for different hardware platforms
#
# The device manager coordinates hardware resources and makes them available to
# the rest of the system through a unified interface.

#--------------------------------------
#       IMPORTS
#--------------------------------------

import os
import sys
import logging
import time
from enum import Enum
from pathlib import Path

# Local imports
from audio_led.hardware.led_controller import (
    LEDStripType, LEDInterface, LEDLayout, ColorOrder, 
    LEDController, NeoPixelController, DotStarController, 
    RPiWS281xController, VirtualLEDController, DummyLEDController,
    LEDManager
)

# Configure module logger
logger = logging.getLogger(__name__)

# Optional imports based on environment
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    logger.warning("PyAudio not available")

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    logger.warning("Sounddevice not available")

try:
    import board
    import neopixel
    HAS_NEOPIXEL = True
except ImportError:
    HAS_NEOPIXEL = False
    logger.warning("NeoPixel library not available")

try:
    import RPi.GPIO as GPIO
    HAS_RPI_GPIO = True
except ImportError:
    HAS_RPI_GPIO = False
    logger.warning("RPi.GPIO library not available")

#--------------------------------------
#       CONSTANTS
#--------------------------------------

# Device types
class DeviceType(Enum):
    AUDIO_INPUT = "audio_input"
    AUDIO_OUTPUT = "audio_output"
    LED = "led"
    DISPLAY = "display"
    UNKNOWN = "unknown"

# Default device names
DEFAULT_DEVICE_NAMES = {
    DeviceType.AUDIO_INPUT: "Default Microphone",
    DeviceType.AUDIO_OUTPUT: "Default Speaker",
    DeviceType.LED: "Default LED Controller",
    DeviceType.DISPLAY: "Default Display"
}

#--------------------------------------
#       FUNCTIONS
#--------------------------------------

def detect_devices():
    """
    Detect available hardware devices.
    
    Returns:
        dict: Dictionary of available devices by type
    """
    devices = {
        'input': [],
        'output': [],
        'led': []
    }
    
    # Detect audio input devices
    if HAS_PYAUDIO:
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info.get('maxInputChannels', 0) > 0:
                    devices['input'].append({
                        'id': i,
                        'name': device_info.get('name', f"Device {i}"),
                        'channels': device_info.get('maxInputChannels', 1),
                        'sample_rate': int(device_info.get('defaultSampleRate', 44100)),
                        'api': 'pyaudio'
                    })
            p.terminate()
        except Exception as e:
            logger.error(f"Error detecting PyAudio devices: {e}")
    
    elif HAS_SOUNDDEVICE:
        try:
            device_list = sd.query_devices()
            for i, device in enumerate(device_list):
                if device.get('max_input_channels', 0) > 0:
                    devices['input'].append({
                        'id': i,
                        'name': device.get('name', f"Device {i}"),
                        'channels': device.get('max_input_channels', 1),
                        'sample_rate': int(device.get('default_samplerate', 44100)),
                        'api': 'sounddevice'
                    })
        except Exception as e:
            logger.error(f"Error detecting Sounddevice devices: {e}")
    
    # Detect audio output devices (similar to input detection)
    if HAS_PYAUDIO:
        try:
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info.get('maxOutputChannels', 0) > 0:
                    devices['output'].append({
                        'id': i,
                        'name': device_info.get('name', f"Device {i}"),
                        'channels': device_info.get('maxOutputChannels', 1),
                        'sample_rate': int(device_info.get('defaultSampleRate', 44100)),
                        'api': 'pyaudio'
                    })
            p.terminate()
        except Exception as e:
            logger.error(f"Error detecting PyAudio output devices: {e}")
    
    # Detect LED devices
    # NeoPixel
    if HAS_NEOPIXEL:
        devices['led'].append({
            'id': 'neopixel',
            'name': 'NeoPixel RGB LEDs',
            'type': LEDStripType.WS2812B.value,
            'interface': LEDInterface.GPIO.value
        })
    
    # Raspberry Pi GPIO
    if HAS_RPI_GPIO:
        devices['led'].append({
            'id': 'rpi_gpio',
            'name': 'Raspberry Pi GPIO (PWM)',
            'type': 'pwm',
            'interface': LEDInterface.PWM.value
        })
    
    # Always add virtual LED controller for testing/development
    devices['led'].append({
        'id': 'virtual',
        'name': 'Virtual LED Display',
        'type': LEDStripType.VIRTUAL.value,
        'interface': LEDInterface.VIRTUAL.value
    })
    
    # Always add dummy LED controller as fallback
    devices['led'].append({
        'id': 'dummy',
        'name': 'Dummy LED Controller',
        'type': LEDStripType.DUMMY.value,
        'interface': LEDInterface.VIRTUAL.value
    })
    
    return devices

#--------------------------------------
#       CLASSES
#--------------------------------------

class DeviceManager:
    """
    Manage hardware devices for the Audio LED Visualization System.
    
    This class provides a unified interface for interacting with hardware devices,
    handling detection, initialization, and configuration of audio and LED devices.
    """
    
    def __init__(self, env_config):
        """
        Initialize the device manager.
        
        Args:
            env_config (dict): Environment configuration
        """
        self.config = env_config.get('config', {})
        self.env_info = env_config.get('env_info', {})
        self.capabilities = self.env_info.get('capabilities', {})
        
        # Hardware configuration
        self.hw_config = self.config.get('hardware', {})
        
        # Initialize device collections
        self.available_devices = detect_devices()
        self.active_devices = {
            DeviceType.AUDIO_INPUT: None,
            DeviceType.AUDIO_OUTPUT: None,
            DeviceType.LED: None,
            DeviceType.DISPLAY: None
        }
        
        # LED manager for controlling LED devices
        self.led_manager = None
        
        logger.info("Device manager initialized")
    
    def get_available_devices(self, device_type=None):
        """
        Get available devices of a specific type.
        
        Args:
            device_type (DeviceType, optional): Type of device to get
            
        Returns:
            list: List of available devices
        """
        if device_type == DeviceType.AUDIO_INPUT:
            return self.available_devices.get('input', [])
        elif device_type == DeviceType.AUDIO_OUTPUT:
            return self.available_devices.get('output', [])
        elif device_type == DeviceType.LED:
            return self.available_devices.get('led', [])
        else:
            return self.available_devices
    
    def initialize_led_controller(self):
        """
        Initialize the LED controller based on configuration.
        
        Returns:
            LEDController: Initialized LED controller
        """
        led_type = self.hw_config.get('led_type', 'auto')
        
        # Auto-detect LED type if set to auto
        if led_type == 'auto':
            if HAS_NEOPIXEL:
                led_type = LEDStripType.WS2812B.value
            elif HAS_RPI_GPIO:
                led_type = 'pwm'
            else:
                led_type = LEDStripType.VIRTUAL.value
        
        # Create an LED manager with the appropriate controller
        self.led_manager = LEDManager(self.config)
        success = self.led_manager.start()
        
        if success:
            logger.info(f"LED controller initialized: {led_type}")
            self.active_devices[DeviceType.LED] = self.led_manager
            return self.led_manager
        else:
            logger.error("Failed to initialize LED controller")
            return None
    
    def get_led_manager(self):
        """
        Get the LED manager instance.
        
        Returns:
            LEDManager: LED manager instance
        """
        if self.led_manager is None:
            self.initialize_led_controller()
        
        return self.led_manager
    
    def close(self):
        """
        Close all active devices and free resources.
        
        Returns:
            bool: True if successful, False otherwise
        """
        success = True
        
        # Close LED manager if initialized
        if self.led_manager:
            try:
                self.led_manager.stop()
                logger.info("LED manager closed")
            except Exception as e:
                logger.error(f"Error closing LED manager: {e}")
                success = False
            
            self.led_manager = None
        
        logger.info("Device manager closed")
        return success

# Test the module if run directly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Detect available devices
    print("Detecting available devices...")
    devices = detect_devices()
    
    print("\nAudio Input Devices:")
    for i, device in enumerate(devices['input']):
        print(f"  [{i}] {device['name']} ({device['api']})")
    
    print("\nAudio Output Devices:")
    for i, device in enumerate(devices['output']):
        print(f"  [{i}] {device['name']} ({device['api']})")
    
    print("\nLED Devices:")
    for i, device in enumerate(devices['led']):
        print(f"  [{i}] {device['name']} ({device['type']})")
    
    # Create a simple environment configuration for testing
    env_config = {
        'config': {
            'hardware': {
                'led_type': 'auto',
                'led_count': 60,
                'led_pin': 18,
                'brightness': 0.5
            }
        },
        'env_info': {
            'system_type': 'unix',
            'capabilities': {
                'led_control': True,
                'audio_input': True
            }
        }
    }
    
    # Initialize device manager
    print("\nInitializing device manager...")
    dev_manager = DeviceManager(env_config)
    
    # Initialize LED controller
    print("Initializing LED controller...")
    led_manager = dev_manager.initialize_led_controller()
    
    if led_manager:
        print("LED controller initialized successfully")
        
        # Test LED controller with a simple pattern
        print("Testing LED controller with a simple pattern...")
        led_manager.set_all_pixels((255, 0, 0))  # Red
        time.sleep(1)
        led_manager.set_all_pixels((0, 255, 0))  # Green
        time.sleep(1)
        led_manager.set_all_pixels((0, 0, 255))  # Blue
        time.sleep(1)
        led_manager.set_all_pixels((0, 0, 0))    # Off
    
    # Close device manager
    print("\nClosing device manager...")
    dev_manager.close()
    
    print("Done!") 