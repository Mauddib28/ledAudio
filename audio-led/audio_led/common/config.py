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

# This module handles configuration management for the Audio LED Visualization System.
# It provides functions to load, save, and validate configuration settings from YAML files,
# as well as providing sensible defaults for all settings.

#--------------------------------------
#       IMPORTS
#--------------------------------------

import os
import sys
import logging
from pathlib import Path

# Try to import yaml, fallback to a simple implementation if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logging.warning("PyYAML not available, using simple config parser")

#--------------------------------------
#       CONSTANTS
#--------------------------------------

# Default configuration settings
DEFAULT_CONFIG = {
    'audio': {
        'input_source': 'auto',  # 'auto', 'microphone', 'file', or specific device/file path
        'input_format': 'auto',  # 'auto', 'wav', 'mp3', 'raw'
        'sample_rate': 44100,    # Default sample rate in Hz
        'channels': 2,           # Default channels (2 for stereo)
        'chunk_size': 2048,      # Default chunk size for processing
        'output_device': 'auto', # 'auto', 'default', or specific device
        'play_audio': True,      # Whether to play audio through speakers
        'resample': True,        # Whether to automatically resample audio when needed
    },
    'processing': {
        'fft_enabled': True,     # Whether to perform FFT
        'window_type': 'blackman', # Window function for FFT ('blackman', 'hamming', 'hann')
        'frequency_range': [20, 20000], # Min and max frequencies to analyze
        'bands': 64,             # Number of frequency bands
        'smoothing': 0.5,        # Smoothing factor (0-1)
    },
    'visual': {
        'output_method': 'auto', # 'auto', 'leds', 'file', 'display'
        'output_file': None,     # Path for output file (if method is 'file')
        'color_mode': 'spectrum', # 'spectrum', 'intensity', 'custom'
        'custom_colors': [],     # List of RGB tuples for custom color schemes
        'brightness': 255,       # LED brightness (0-255)
        'refresh_rate': 30,      # Target refresh rate for visualization (Hz)
    },
    'hardware': {
        'led_type': 'auto',      # 'auto', 'ws2812', 'pi_pwm', 'arduino'
        'led_count': 60,         # Number of LEDs
        'led_pin': 18,           # Data pin for LED strip
        'red_pin': 17,           # PWM pin for red channel (if using separate RGB)
        'green_pin': 22,         # PWM pin for green channel
        'blue_pin': 24,          # PWM pin for blue channel
        'invert_logic': False,   # Whether to invert output logic
    },
    'system': {
        'log_level': 'info',     # 'debug', 'info', 'warning', 'error'
        'gui_enabled': 'auto',   # 'auto', True, False
        'dependencies_check': True, # Whether to check and install dependencies
    }
}

#--------------------------------------
#       FUNCTIONS
#--------------------------------------

def get_default_config():
    """
    Get the default configuration.
    
    Returns:
        dict: Default configuration dictionary
    """
    return DEFAULT_CONFIG.copy()

def simple_parse_yaml(file_path):
    """
    A very simple YAML parser for basic configurations when PyYAML is not available.
    Only supports a limited subset of YAML syntax.
    
    Args:
        file_path (str): Path to YAML file
        
    Returns:
        dict: Parsed configuration
    """
    config = {}
    current_section = config
    section_stack = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
                
            # Check indentation level (count leading spaces)
            indent = len(line) - len(line.lstrip())
            indent_level = indent // 2  # Assume 2 spaces per level
            
            # Adjust section stack based on indentation
            while len(section_stack) > indent_level:
                section_stack.pop()
                
            # Navigate to current section based on stack
            current_section = config
            for section in section_stack:
                current_section = current_section[section]
            
            # Parse the line
            if ':' in line:
                key, value = [x.strip() for x in line.split(':', 1)]
                
                # Check if this is a new section
                if not value:
                    current_section[key] = {}
                    section_stack.append(key)
                else:
                    # Convert value to appropriate type
                    if value.lower() == 'true':
                        current_section[key] = True
                    elif value.lower() == 'false':
                        current_section[key] = False
                    elif value.lower() == 'null' or value.lower() == 'none':
                        current_section[key] = None
                    elif value.isdigit():
                        current_section[key] = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        current_section[key] = float(value)
                    else:
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        current_section[key] = value
    
    return config

def load_config(file_path):
    """
    Load configuration from a YAML file.
    
    Args:
        file_path (str): Path to configuration file
        
    Returns:
        dict: Loaded configuration merged with defaults
    """
    config = get_default_config()
    
    try:
        if HAS_YAML:
            with open(file_path, 'r') as f:
                user_config = yaml.safe_load(f)
        else:
            user_config = simple_parse_yaml(file_path)
            
        # Deep merge user config into defaults
        deep_merge(config, user_config)
        
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        logging.warning("Using default configuration")
    
    # Validate the merged configuration
    validate_config(config)
    
    return config

def deep_merge(base, update):
    """
    Recursively merge two dictionaries.
    
    Args:
        base (dict): Base dictionary to merge into
        update (dict): Dictionary with updates to apply
        
    Returns:
        dict: Merged dictionary
    """
    if not isinstance(update, dict):
        return update
    
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    
    return base

def validate_config(config):
    """
    Validate configuration values and fix any issues.
    
    Args:
        config (dict): Configuration to validate
        
    Returns:
        dict: Validated configuration
    """
    # Validate audio configuration
    if config['audio']['sample_rate'] <= 0:
        logging.warning("Invalid sample rate, using default")
        config['audio']['sample_rate'] = DEFAULT_CONFIG['audio']['sample_rate']
        
    if config['audio']['channels'] <= 0:
        logging.warning("Invalid channel count, using default")
        config['audio']['channels'] = DEFAULT_CONFIG['audio']['channels']
        
    if config['audio']['chunk_size'] <= 0:
        logging.warning("Invalid chunk size, using default")
        config['audio']['chunk_size'] = DEFAULT_CONFIG['audio']['chunk_size']
    
    # Validate processing configuration
    if config['processing']['bands'] <= 0:
        logging.warning("Invalid band count, using default")
        config['processing']['bands'] = DEFAULT_CONFIG['processing']['bands']
        
    if not 0 <= config['processing']['smoothing'] <= 1:
        logging.warning("Invalid smoothing factor, using default")
        config['processing']['smoothing'] = DEFAULT_CONFIG['processing']['smoothing']
    
    # Validate visual configuration
    if not 0 <= config['visual']['brightness'] <= 255:
        logging.warning("Invalid brightness, using default")
        config['visual']['brightness'] = DEFAULT_CONFIG['visual']['brightness']
        
    if config['visual']['refresh_rate'] <= 0:
        logging.warning("Invalid refresh rate, using default")
        config['visual']['refresh_rate'] = DEFAULT_CONFIG['visual']['refresh_rate']
    
    # Validate hardware configuration
    if config['hardware']['led_count'] <= 0:
        logging.warning("Invalid LED count, using default")
        config['hardware']['led_count'] = DEFAULT_CONFIG['hardware']['led_count']
    
    return config

def save_config(config, file_path):
    """
    Save configuration to a YAML file.
    
    Args:
        config (dict): Configuration to save
        file_path (str): Path to save to
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    if not HAS_YAML:
        logging.error("Cannot save configuration without PyYAML")
        return False
    
    try:
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        logging.error(f"Error saving configuration: {e}")
        return False

def create_default_config_file(file_path=None):
    """
    Create a default configuration file.
    
    Args:
        file_path (str, optional): Path to save the configuration file
        
    Returns:
        str: Path to the created configuration file
    """
    if file_path is None:
        file_path = os.path.join(os.getcwd(), 'config.yaml')
    
    if not HAS_YAML:
        logging.error("Cannot create configuration file without PyYAML")
        return None
    
    try:
        with open(file_path, 'w') as f:
            yaml.dump(get_default_config(), f, default_flow_style=False)
        logging.info(f"Default configuration saved to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error creating default configuration: {e}")
        return None

# If run directly, create a default configuration file
if __name__ == "__main__":
    if len(sys.argv) > 1:
        create_default_config_file(sys.argv[1])
    else:
        create_default_config_file() 