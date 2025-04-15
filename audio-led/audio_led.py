#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####
# Created on:        2025-04-15
# Created by:        AI Assistant
# Last modified on:  2025-04-15
####

#--------------------------------------
#   BACKGROUND RESEARCH HISTORY
#--------------------------------------

# This code is a modular reimplementation of musicReact.py and psynesthesia_modified.py
# It aims to create a system that visualizes audio input through LED light patterns.
#
# Original Credits:
# - musicReact.py: Based on work by David Ordnung and Paul A. Wortman
# - psynesthesia_modified.py: Based on work by rho-bit
# 
# The system processes audio input in real-time, extracts features such as frequency and amplitude,
# and translates these features into visually appealing LED light patterns.
#
# This main script serves as the entry point and orchestrator for the entire system,
# handling command-line arguments, environment detection, and high-level coordination
# between audio input, processing, and visualization modules.

#--------------------------------------
#       IMPORTS
#--------------------------------------

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Local module imports
# Import these after environment setup to handle missing dependencies gracefully
try:
    from audio_led.common import config, environment
    from audio_led.audio import input_handler, processor
    from audio_led.visual import rgb_converter, output_handler
    from audio_led.hardware import device_manager
    from audio_led.utils import installer, logger
except ImportError:
    # Add the parent directory to the system path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from audio_led.common import config, environment
    from audio_led.audio import input_handler, processor
    from audio_led.visual import rgb_converter, output_handler
    from audio_led.hardware import device_manager
    from audio_led.utils import installer, logger

#--------------------------------------
#       CONSTANTS
#--------------------------------------

VERSION = "1.0.0"
DEFAULT_CONFIG_FILE = "config.yaml"

#--------------------------------------
#       FUNCTIONS
#--------------------------------------

def parse_arguments():
    """
    Parse command-line arguments for the application.
    
    Returns:
        argparse.Namespace: The parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Audio LED Visualization System')
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG_FILE, 
                        help=f'Path to configuration file (default: {DEFAULT_CONFIG_FILE})')
    parser.add_argument('-i', '--input', type=str, 
                        help='Audio input source (file path or device)')
    parser.add_argument('-o', '--output', type=str, 
                        help='Output method (leds, file, display)')
    parser.add_argument('-d', '--debug', action='store_true', 
                        help='Enable debug logging')
    parser.add_argument('-g', '--gui', action='store_true', 
                        help='Launch with graphical user interface')
    parser.add_argument('-v', '--version', action='version', 
                        version=f'Audio LED Visualization System v{VERSION}')
    parser.add_argument('--detect', action='store_true', 
                        help='Detect and list available audio input/output devices')
    return parser.parse_args()

def setup_environment(args):
    """
    Initialize the environment based on detected hardware and configuration.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments
        
    Returns:
        dict: Environment configuration
    """
    # Initialize logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setup_logging(log_level)
    logging.info("Starting Audio LED Visualization System")
    
    # Detect environment
    env_info = environment.detect_environment()
    logging.info(f"Detected environment: {env_info['system_type']}")
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logging.warning(f"Configuration file {args.config} not found, using defaults")
        cfg = config.get_default_config()
    else:
        cfg = config.load_config(args.config)
    
    # Install dependencies if needed
    if not env_info.get('dependencies_installed', False):
        if installer.check_and_install_dependencies(env_info['system_type']):
            logging.info("Dependencies installed successfully")
        else:
            logging.warning("Some dependencies could not be installed")
    
    # Override config with command-line arguments
    if args.input:
        cfg['audio']['input_source'] = args.input
    if args.output:
        cfg['visual']['output_method'] = args.output
    
    # Return combined configuration
    return {
        'env_info': env_info,
        'config': cfg,
        'args': args
    }

def run_headless(env_config):
    """
    Run the system in headless mode (command-line only).
    
    Args:
        env_config (dict): Environment configuration
    """
    logging.info("Running in headless mode")
    
    # Set colorful visualization mode explicitly
    if 'config' in env_config and 'visual' in env_config['config']:
        env_config['config']['visual']['color_mode'] = 'colorful'
    
    # Initialize hardware
    hw_manager = device_manager.DeviceManager(env_config)
    
    # Initialize audio input with prompt for source
    audio_input = input_handler.AudioInputHandler(env_config)
    
    # Make sure we have a valid input source before continuing
    if audio_input.input_type is None:
        logging.error("No audio input source selected, exiting")
        return
    
    # Initialize audio processor
    audio_proc = processor.AudioProcessor(env_config)
    audio_proc.audio_input = audio_input
    
    # Initialize RGB converter
    rgb_conv = rgb_converter.RGBConverter(env_config)
    
    # Initialize output handler
    output = output_handler.OutputHandler(env_config, hw_manager)
    
    # Start processing loop
    try:
        logging.info("Starting audio processing loop")
        while True:
            # Get audio data
            audio_data = audio_input.get_audio_chunk()
            if audio_data is None:
                break
            
            # Process audio data
            processed_data = audio_proc.process(audio_data)
            
            # Convert to RGB
            rgb_values = rgb_conv.convert(processed_data)
            
            # Output the RGB values
            output.update(rgb_values)
            
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, shutting down")
    finally:
        # Cleanup
        audio_input.close()
        output.close()
        logging.info("System shutdown complete")

def run_with_gui(env_config):
    """
    Run the system with a graphical user interface.
    
    Args:
        env_config (dict): Environment configuration
    """
    logging.info("Running with GUI")
    
    # Import GUI modules here to avoid dependencies when running headless
    try:
        from audio_led.utils import gui
        app = gui.Application(env_config)
        app.run()
    except ImportError as e:
        logging.error(f"Failed to import GUI modules: {e}")
        logging.info("Falling back to headless mode")
        run_headless(env_config)

def main():
    """
    Main entry point for the application.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up environment
    env_config = setup_environment(args)
    
    # If the user wants to detect devices, do that and exit
    if args.detect:
        devices = device_manager.detect_devices()
        print("\nDetected Audio Input Devices:")
        for i, device in enumerate(devices['input']):
            print(f"  [{i}] {device['name']}")
        
        print("\nDetected Audio Output Devices:")
        for i, device in enumerate(devices['output']):
            print(f"  [{i}] {device['name']}")
        
        print("\nDetected LED Devices:")
        for i, device in enumerate(devices['led']):
            print(f"  [{i}] {device['name']} ({device['type']})")
        return
    
    # Run in either GUI or headless mode
    if args.gui and env_config['env_info']['system_type'] != 'embedded':
        run_with_gui(env_config)
    else:
        run_headless(env_config)

if __name__ == "__main__":
    main() 