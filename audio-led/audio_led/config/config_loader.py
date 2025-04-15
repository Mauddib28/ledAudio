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

# This module handles configuration loading for the Audio LED Visualization System.
# It provides functionality to:
# - Detect the runtime environment (hardware platform, capabilities)
# - Load configuration from files (YAML, JSON, etc.)
# - Provide default configurations for different environments
# - Validate configurations against a schema
# - Merge configurations from multiple sources
#
# The configuration is used by all other modules to adapt to the
# current environment and user preferences.

#--------------------------------------
#       IMPORTS
#--------------------------------------

import os
import sys
import json
import logging
import platform
import subprocess
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logging.warning("YAML support not available, falling back to JSON")

# Configure module logger
logger = logging.getLogger(__name__)

#--------------------------------------
#       CONSTANTS
#--------------------------------------

# Environment types
ENV_RASPBERRY_PI = "raspberry_pi"
ENV_PICO_W = "pico_w"
ENV_UNIX = "unix"
ENV_WINDOWS = "windows"
ENV_UNKNOWN = "unknown"

# Default configuration paths
DEFAULT_CONFIG_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "defaults"
USER_CONFIG_DIR = Path.home() / ".config" / "audio-led"

# Default configuration files
DEFAULT_CONFIG_FILE = "config.yaml" if HAS_YAML else "config.json"

# Default configuration by environment
DEFAULT_CONFIG = {
    ENV_RASPBERRY_PI: {
        "audio": {
            "input_device": "default",
            "input_method": "microphone",
            "sample_rate": 44100,
            "channels": 1,
            "chunk_size": 1024,
            "overlap": 0.5
        },
        "visual": {
            "output_method": "led_pwm",
            "color_mode": "spectrum",
            "refresh_rate": 30,
            "led_smoothing": 0.5,
            "display_width": 800,
            "display_height": 600
        },
        "hardware": {
            "led_type": "rgb",
            "led_count": 1,
            "led_pin": 18,
            "red_pin": 17,
            "green_pin": 22,
            "blue_pin": 24
        },
        "system": {
            "log_level": "INFO",
            "headless": True,
            "debug_mode": False
        }
    },
    ENV_PICO_W: {
        "audio": {
            "input_device": "default",
            "input_method": "i2s_microphone",
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 512,
            "overlap": 0.5
        },
        "visual": {
            "output_method": "led_pwm",
            "color_mode": "spectrum",
            "refresh_rate": 20,
            "led_smoothing": 0.7
        },
        "hardware": {
            "led_type": "rgb",
            "led_count": 1,
            "red_pin": 17,
            "green_pin": 22,
            "blue_pin": 16
        },
        "system": {
            "log_level": "WARNING",
            "headless": True,
            "debug_mode": False
        }
    },
    ENV_UNIX: {
        "audio": {
            "input_device": "default",
            "input_method": "microphone",
            "sample_rate": 44100,
            "channels": 2,
            "chunk_size": 1024,
            "overlap": 0.5
        },
        "visual": {
            "output_method": "display",
            "color_mode": "spectrum",
            "refresh_rate": 60,
            "led_smoothing": 0.3,
            "display_width": 800,
            "display_height": 600
        },
        "hardware": {
            "led_type": "none",
            "led_count": 0
        },
        "system": {
            "log_level": "INFO",
            "headless": False,
            "debug_mode": True
        }
    },
    ENV_WINDOWS: {
        "audio": {
            "input_device": "default",
            "input_method": "microphone",
            "sample_rate": 44100,
            "channels": 2,
            "chunk_size": 1024,
            "overlap": 0.5
        },
        "visual": {
            "output_method": "display",
            "color_mode": "spectrum",
            "refresh_rate": 60,
            "led_smoothing": 0.3,
            "display_width": 800,
            "display_height": 600
        },
        "hardware": {
            "led_type": "none",
            "led_count": 0
        },
        "system": {
            "log_level": "INFO",
            "headless": False,
            "debug_mode": True
        }
    }
}

#--------------------------------------
#   ENVIRONMENT DETECTION
#--------------------------------------

def detect_environment():
    """
    Detect the current runtime environment.
    
    Returns:
        dict: Environment information
    """
    env_info = {
        "system_type": ENV_UNKNOWN,
        "os_name": platform.system(),
        "os_version": platform.release(),
        "python_version": platform.python_version(),
        "capabilities": {
            "gpio": False,
            "display": False,
            "audio_input": False,
            "audio_output": False,
            "network": False
        }
    }
    
    # Detect platform type
    if env_info["os_name"] == "Linux":
        # Check if running on Raspberry Pi
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'raspberry pi' in model.lower():
                    env_info["system_type"] = ENV_RASPBERRY_PI
                    env_info["model"] = model.strip('\0')
        except (FileNotFoundError, IOError):
            # Not a Raspberry Pi or can't read the model file
            env_info["system_type"] = ENV_UNIX
    
        # Check for RP2040 (Pico W) under MicroPython
        if hasattr(sys, 'implementation') and sys.implementation.name == 'micropython':
            try:
                import machine
                if hasattr(machine, 'unique_id'):
                    env_info["system_type"] = ENV_PICO_W
            except ImportError:
                pass
    
    elif env_info["os_name"] == "Windows":
        env_info["system_type"] = ENV_WINDOWS
    
    # Detect capabilities
    env_info["capabilities"] = detect_capabilities(env_info["system_type"])
    
    return env_info

def detect_capabilities(system_type):
    """
    Detect hardware and software capabilities.
    
    Args:
        system_type (str): System type (raspberry_pi, unix, etc.)
        
    Returns:
        dict: Detected capabilities
    """
    capabilities = {
        "gpio": False,
        "display": False,
        "audio_input": False,
        "audio_output": False,
        "network": False
    }
    
    # GPIO capabilities
    if system_type == ENV_RASPBERRY_PI:
        # Check for GPIO access
        try:
            import RPi.GPIO as GPIO
            capabilities["gpio"] = True
        except ImportError:
            try:
                import pigpio
                capabilities["gpio"] = True
            except ImportError:
                pass
    
    elif system_type == ENV_PICO_W:
        # Check for GPIO access on Pico W
        try:
            import machine
            capabilities["gpio"] = True
        except ImportError:
            pass
    
    # Display capabilities
    try:
        # Check for display capability
        if system_type in [ENV_UNIX, ENV_WINDOWS, ENV_RASPBERRY_PI]:
            # Try to detect a display
            if "DISPLAY" in os.environ or system_type == ENV_WINDOWS:
                capabilities["display"] = True
    except:
        pass
    
    # Audio input capabilities
    try:
        if system_type in [ENV_UNIX, ENV_WINDOWS, ENV_RASPBERRY_PI]:
            # Check for audio input devices
            if system_type in [ENV_UNIX, ENV_RASPBERRY_PI]:
                try:
                    # Try to use ALSA to list audio devices
                    result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
                    if result.returncode == 0 and 'card' in result.stdout:
                        capabilities["audio_input"] = True
                except (FileNotFoundError, subprocess.SubprocessError):
                    # ALSA not available, try another method
                    pass
            
            # Try to import audio libraries
            try:
                import pyaudio
                capabilities["audio_input"] = True
                capabilities["audio_output"] = True
            except ImportError:
                try:
                    import sounddevice
                    capabilities["audio_input"] = True
                    capabilities["audio_output"] = True
                except ImportError:
                    pass
        
        elif system_type == ENV_PICO_W:
            # Check for audio on Pico W
            try:
                from machine import I2S
                capabilities["audio_input"] = True
            except ImportError:
                pass
    except:
        pass
    
    # Network capabilities
    try:
        import socket
        # Try to create a socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        s.close()
        capabilities["network"] = True
    except:
        pass
    
    return capabilities

#--------------------------------------
#   CONFIGURATION HANDLING
#--------------------------------------

class ConfigLoader:
    """
    Configuration loader for Audio LED Visualization System.
    
    This class handles loading, merging, and validating configurations
    from various sources.
    """
    
    def __init__(self, config_file=None, create_if_missing=True):
        """
        Initialize the configuration loader.
        
        Args:
            config_file (str, optional): Path to the configuration file.
                If not specified, the default configuration file will be used.
            create_if_missing (bool, optional): Whether to create the configuration
                file if it doesn't exist.
        """
        self.config_file = config_file
        self.create_if_missing = create_if_missing
        
        # Detect the environment
        self.env_info = detect_environment()
        self.system_type = self.env_info["system_type"]
        
        # Get default configuration for the current environment
        self.default_config = DEFAULT_CONFIG.get(self.system_type, DEFAULT_CONFIG[ENV_UNIX])
        
        # Initialize configuration
        self.config = {}
        
        logger.info(f"ConfigLoader initialized for {self.system_type} environment")
    
    def load_config(self):
        """
        Load the configuration from the specified file.
        
        If the file doesn't exist and create_if_missing is True,
        the default configuration will be saved to the file.
        
        Returns:
            dict: Loaded configuration
        """
        if self.config_file is None:
            # Use default config file location
            if self.system_type == ENV_PICO_W:
                # On Pico W, use a config.py file in the root directory
                self.config_file = "config.py"
            else:
                # On other systems, use the user config directory
                os.makedirs(USER_CONFIG_DIR, exist_ok=True)
                self.config_file = USER_CONFIG_DIR / DEFAULT_CONFIG_FILE
        
        # Load configuration from file
        try:
            config = self._load_from_file(self.config_file)
            logger.info(f"Configuration loaded from {self.config_file}")
        except FileNotFoundError:
            logger.warning(f"Configuration file {self.config_file} not found")
            
            if self.create_if_missing:
                # Create the configuration file with default settings
                logger.info(f"Creating default configuration file at {self.config_file}")
                config = self.default_config.copy()
                self._save_to_file(self.config_file, config)
            else:
                # Use default configuration without saving
                logger.info("Using default configuration")
                config = self.default_config.copy()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            config = self.default_config.copy()
        
        # Merge with default configuration to ensure all required fields are present
        self.config = self._merge_configs(self.default_config, config)
        
        # Validate the configuration
        self._validate_config()
        
        # Add environment information to the configuration
        self.config["env_info"] = self.env_info
        
        return self.config
    
    def _load_from_file(self, file_path):
        """
        Load configuration from a file.
        
        Args:
            file_path (str): Path to the configuration file
            
        Returns:
            dict: Loaded configuration
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file {file_path} not found")
        
        if file_path.suffix == '.yaml' or file_path.suffix == '.yml':
            if not HAS_YAML:
                raise ValueError("YAML support not available")
            
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        
        elif file_path.suffix == '.py':
            # Load Python configuration file (for MicroPython)
            local_vars = {}
            with open(file_path, 'r') as f:
                exec(f.read(), {}, local_vars)
            
            # Extract configuration from local variables
            if 'CONFIG' in local_vars:
                return local_vars['CONFIG']
            else:
                raise ValueError("Python configuration file must define CONFIG variable")
        
        else:
            raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
    
    def _save_to_file(self, file_path, config):
        """
        Save configuration to a file.
        
        Args:
            file_path (str): Path to the configuration file
            config (dict): Configuration to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            os.makedirs(file_path.parent, exist_ok=True)
            
            if file_path.suffix == '.yaml' or file_path.suffix == '.yml':
                if not HAS_YAML:
                    raise ValueError("YAML support not available")
                
                with open(file_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            
            elif file_path.suffix == '.json':
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            elif file_path.suffix == '.py':
                # Save as Python configuration file (for MicroPython)
                with open(file_path, 'w') as f:
                    f.write("# Audio LED Visualization System Configuration\n\n")
                    f.write("CONFIG = ")
                    f.write(repr(config))
                    f.write("\n")
            
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def _merge_configs(self, base_config, override_config):
        """
        Merge two configurations, with override_config taking precedence.
        
        Args:
            base_config (dict): Base configuration
            override_config (dict): Override configuration
            
        Returns:
            dict: Merged configuration
        """
        merged_config = base_config.copy()
        
        # Recursively merge the configurations
        def merge_dicts(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    # Recursively merge dictionaries
                    merge_dicts(base[key], value)
                else:
                    # Override the value
                    base[key] = value
        
        merge_dicts(merged_config, override_config)
        
        return merged_config
    
    def _validate_config(self):
        """
        Validate the configuration.
        
        This method checks that the configuration contains all required fields
        and that the values are of the correct type.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        # Check for required sections
        required_sections = ["audio", "visual", "hardware", "system"]
        
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing configuration section: {section}")
                self.config[section] = self.default_config[section]
        
        # Validate specific fields
        # (This could be extended with more detailed validation)
        
        # Audio settings
        audio_config = self.config["audio"]
        if not isinstance(audio_config.get("sample_rate"), int) or audio_config.get("sample_rate") <= 0:
            logger.warning("Invalid sample_rate, using default")
            audio_config["sample_rate"] = self.default_config["audio"]["sample_rate"]
        
        if not isinstance(audio_config.get("chunk_size"), int) or audio_config.get("chunk_size") <= 0:
            logger.warning("Invalid chunk_size, using default")
            audio_config["chunk_size"] = self.default_config["audio"]["chunk_size"]
        
        # Visual settings
        visual_config = self.config["visual"]
        if not isinstance(visual_config.get("refresh_rate"), (int, float)) or visual_config.get("refresh_rate") <= 0:
            logger.warning("Invalid refresh_rate, using default")
            visual_config["refresh_rate"] = self.default_config["visual"]["refresh_rate"]
        
        # System settings
        system_config = self.config["system"]
        if system_config.get("log_level") not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            logger.warning("Invalid log_level, using default")
            system_config["log_level"] = self.default_config["system"]["log_level"]
    
    def save_config(self):
        """
        Save the current configuration to the specified file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.config_file is None:
            logger.error("No configuration file specified")
            return False
        
        # Create a copy of the configuration without the env_info
        config_to_save = self.config.copy()
        if "env_info" in config_to_save:
            del config_to_save["env_info"]
        
        return self._save_to_file(self.config_file, config_to_save)
    
    def get_config(self):
        """
        Get the current configuration.
        
        Returns:
            dict: Current configuration
        """
        return self.config

# Test the module if run directly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a configuration loader
    loader = ConfigLoader()
    
    # Load the configuration
    config = loader.load_config()
    
    # Print the environment information
    print("\nEnvironment Information:")
    print(f"System Type: {config['env_info']['system_type']}")
    print(f"OS: {config['env_info']['os_name']} {config['env_info']['os_version']}")
    print(f"Python Version: {config['env_info']['python_version']}")
    
    print("\nCapabilities:")
    for capability, available in config['env_info']['capabilities'].items():
        print(f"  {capability}: {'Yes' if available else 'No'}")
    
    # Print the configuration
    print("\nConfiguration:")
    for section in ['audio', 'visual', 'hardware', 'system']:
        print(f"\n{section.upper()}:")
        for key, value in config[section].items():
            print(f"  {key}: {value}") 