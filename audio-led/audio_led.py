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
import signal
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
DEFAULT_RUNTIME_LIMIT = 3600  # Default runtime limit (1 hour) to prevent endless loops

# Supported input and output types
SUPPORTED_INPUT_TYPES = ['microphone', 'file']
SUPPORTED_OUTPUT_TYPES = ['led_pwm', 'led_strip', 'display', 'file', 'none']

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
    
    # Input/output options with more detail
    parser.add_argument('-i', '--input', type=str, metavar='SOURCE',
                        help=f'Audio input source: {", ".join(SUPPORTED_INPUT_TYPES)} or a file path')
    parser.add_argument('-o', '--output', type=str, metavar='METHOD',
                        help=f'Output method: {", ".join(SUPPORTED_OUTPUT_TYPES)}')
    
    # Other options
    parser.add_argument('-d', '--debug', action='store_true', 
                        help='Enable debug logging')
    parser.add_argument('-g', '--gui', action='store_true', 
                        help='Launch with graphical user interface')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable all graphical display output, force file output instead')
    parser.add_argument('-t', '--timeout', type=int, default=DEFAULT_RUNTIME_LIMIT,
                        help=f'Maximum runtime in seconds (default: {DEFAULT_RUNTIME_LIMIT} seconds)')
    parser.add_argument('-v', '--version', action='version', 
                        version=f'Audio LED Visualization System v{VERSION}')
    parser.add_argument('--detect', action='store_true', 
                        help='Detect and list available audio input/output devices')
    
    return parser.parse_args()

def prompt_for_input():
    """
    Prompt the user to select an audio input source.
    
    Returns:
        str: Selected audio input source
    """
    print("\nSelect audio input source:")
    print("  [0] Microphone input")
    print("  [1] Audio file")
    
    while True:
        try:
            choice = int(input("Enter selection number: "))
            if choice == 0:
                return "microphone"
            elif choice == 1:
                file_path = input("Enter audio file path: ")
                if os.path.isfile(file_path):
                    return file_path
                else:
                    print(f"File not found: {file_path}")
            else:
                print("Invalid selection. Please enter 0 or 1.")
        except ValueError:
            print("Please enter a valid number")

def prompt_for_output():
    """
    Prompt the user to select an output method.
    
    Returns:
        str: Selected output method
    """
    print("\nSelect output method:")
    print("  [0] LED PWM (direct GPIO control of RGB LEDs)")
    print("  [1] LED Strip (WS2812/NeoPixel)")
    print("  [2] Display (on-screen visualization)")
    print("  [3] File (write to text file)")
    print("  [4] None (for testing)")
    
    while True:
        try:
            choice = int(input("Enter selection number: "))
            if 0 <= choice <= 4:
                return SUPPORTED_OUTPUT_TYPES[choice]
            else:
                print(f"Invalid selection. Please enter a number between 0 and 4.")
        except ValueError:
            print("Please enter a valid number")

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
    
    # Check and handle input source
    if args.input:
        cfg['audio']['input_source'] = args.input
        logging.info(f"Using specified input source: {args.input}")
    else:
        # Input source not specified via command line
        logging.info("No input source specified via command line")
        cfg['audio']['input_source'] = None  # Will prompt later
    
    # Check and handle output method
    if args.output:
        cfg['visual']['output_method'] = args.output
        logging.info(f"Using specified output method: {args.output}")
    else:
        # Output method not specified via command line
        logging.info("No output method specified via command line")
        if args.no_display:
            cfg['visual']['output_method'] = 'file'
            logging.info("Forcing file output due to --no-display flag")
        else:
            cfg['visual']['output_method'] = None  # Will prompt later
    
    # Force file output if no-display is specified
    if args.no_display and cfg['visual']['output_method'] == 'display':
        logging.info("Forcing file output due to --no-display flag")
        cfg['visual']['output_method'] = 'file'
        
    # Ensure we don't use display capabilities if no-display is specified
    if args.no_display:
        if 'capabilities' in env_info:
            env_info['capabilities']['display'] = False
    
    # Return combined configuration
    return {
        'env_info': env_info,
        'config': cfg,
        'args': args
    }

def signal_handler(sig, frame):
    """
    Handle interrupt signals to gracefully exit the application
    """
    logging.info("Signal received, shutting down")
    sys.exit(0)

def run_headless(env_config):
    """
    Run the system in headless mode (command-line only).
    
    Args:
        env_config (dict): Environment configuration
    """
    logging.info("Running in headless mode")
    
    # Register signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Get runtime limit from config
    runtime_limit = env_config['args'].timeout
    
    # Check if we need to prompt for input source
    if env_config['config']['audio']['input_source'] is None:
        print("\n=== Audio LED Visualization System ===")
        print("Command-line argument for input source is missing.")
        env_config['config']['audio']['input_source'] = prompt_for_input()
        logging.info(f"User selected input source: {env_config['config']['audio']['input_source']}")
    
    # Check if we need to prompt for output method
    if env_config['config']['visual']['output_method'] is None:
        print("\nCommand-line argument for output method is missing.")
        env_config['config']['visual']['output_method'] = prompt_for_output()
        logging.info(f"User selected output method: {env_config['config']['visual']['output_method']}")
    
    # Ensure we're using file output in headless mode if display was selected
    if env_config['config']['visual']['output_method'] == 'display' and env_config['args'].no_display:
        logging.info("Headless mode with --no-display detected, forcing file output instead of display")
        env_config['config']['visual']['output_method'] = 'file'
        
    # Set colorful visualization mode
    if 'config' in env_config and 'visual' in env_config['config']:
        env_config['config']['visual']['color_mode'] = 'colorful'
    
    # Initialize hardware
    hw_manager = device_manager.DeviceManager(env_config)
    
    # Initialize audio input
    print("\n=== Audio input initialization ===")
    audio_input = input_handler.AudioInputHandler(env_config)
    
    # Make sure we have a valid input source before continuing
    if audio_input.input_type is None:
        logging.error("No audio input source selected, exiting")
        print("\nNo audio input source was selected. Exiting.")
        return
    
    # Initialize audio processor
    audio_proc = processor.AudioProcessor(env_config)
    audio_proc.audio_input = audio_input
    
    # Initialize RGB converter
    rgb_conv = rgb_converter.RGBConverter(env_config)
    
    # Initialize output handler
    output = output_handler.OutputHandler(env_config, hw_manager)
    
    # Print instructions for user interaction
    print("\nAudio LED Visualization running in headless mode")
    print(f"Using input type: {audio_input.input_type}")
    print(f"Output method: {env_config['config']['visual']['output_method']}")
    print("Press Ctrl+C to stop the visualization")
    print(f"Visualization will automatically stop after {runtime_limit} seconds\n")
    
    # Start processing loop
    start_time = time.time()
    loop_count = 0
    no_data_count = 0
    
    try:
        logging.info("Starting audio processing loop")
        
        while True:
            # Check if we've exceeded the runtime limit
            if time.time() - start_time > runtime_limit:
                logging.info(f"Runtime limit of {runtime_limit} seconds reached, shutting down")
                break
            
            # Get audio data
            audio_data = audio_input.get_audio_chunk()
            
            # If no audio data is available, handle accordingly
            if audio_data is None:
                no_data_count += 1
                if no_data_count >= 3:  # After 3 attempts with no data
                    if audio_input.is_file:
                        # For file input, end of file means we're done
                        logging.info("End of audio file reached, shutting down")
                        print("End of audio file reached. Exiting.")
                        break
                    else:
                        # For microphone, wait and retry
                        logging.warning("No audio data received from microphone")
                        time.sleep(0.5)
                        no_data_count = 0
                continue
            
            # Reset the no data counter if we got data
            no_data_count = 0
            
            # Process audio data
            processed_data = audio_proc.process(audio_data)
            
            # Ensure we have valid processed data
            if processed_data is None:
                logging.warning("No processed audio data available")
                time.sleep(0.1)
                continue
                
            # Check if processed data has valid content
            has_valid_data = False
            try:
                # Check if any key in processed_data has non-zero data
                for key, value in processed_data.items():
                    if hasattr(value, 'any'):  # For numpy arrays
                        if value.any():
                            has_valid_data = True
                            break
                    elif hasattr(value, '__len__'):  # For lists, tuples, etc.
                        if len(value) > 0 and any(v != 0 for v in value):
                            has_valid_data = True
                            break
                    elif value:  # For scalar values
                        has_valid_data = True
                        break
                
                if not has_valid_data:
                    logging.warning("Processed data contains no usable values")
                    time.sleep(0.1)
                    continue
            except Exception as e:
                logging.warning(f"Error validating processed data: {e}")
                # Continue anyway to avoid stopping the flow
            
            # Convert to RGB
            rgb_values = rgb_conv.convert(processed_data)
            
            # Output the RGB values
            output.update(rgb_values)
            
            # Increment the loop count
            loop_count += 1
            
            # Every 100 loops, display a heartbeat message
            if loop_count % 100 == 0:
                print(f"Processed {loop_count} audio chunks (running for {int(time.time() - start_time)} seconds)")
            
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, shutting down")
        print("\nVisualization stopped by user.")
    finally:
        # Cleanup
        audio_input.close()
        output.close()
        logging.info("System shutdown complete")
        print("System shutdown complete.")

def run_with_gui(env_config):
    """
    Run the system with a graphical user interface.
    
    Args:
        env_config (dict): Environment configuration
    """
    logging.info("Running with GUI")
    
    # Check if we need to prompt for input source
    if env_config['config']['audio']['input_source'] is None:
        print("\n=== Audio LED Visualization System ===")
        print("Command-line argument for input source is missing.")
        env_config['config']['audio']['input_source'] = prompt_for_input()
        logging.info(f"User selected input source: {env_config['config']['audio']['input_source']}")
    
    # Check if we need to prompt for output method
    if env_config['config']['visual']['output_method'] is None:
        print("\nCommand-line argument for output method is missing.")
        env_config['config']['visual']['output_method'] = prompt_for_output()
        logging.info(f"User selected output method: {env_config['config']['visual']['output_method']}")
    
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
    
    # Check if both input and output are specified
    has_input = env_config['config']['audio']['input_source'] is not None
    has_output = env_config['config']['visual']['output_method'] is not None
    
    if not has_input or not has_output:
        print("\n=== Audio LED Visualization System ===")
        print("Warning: Required command-line arguments are missing.")
        print("You will be prompted to provide the missing information.")
        print("For future runs, use the following arguments:")
        print("  -i SOURCE  Specify audio input source (microphone, file path)")
        print("  -o METHOD  Specify output method (led_pwm, led_strip, display, file, none)")
        print("  --help     Show all available options\n")
    
    # Run in either GUI or headless mode
    if args.gui and env_config['env_info']['system_type'] != 'embedded' and not args.no_display:
        run_with_gui(env_config)
    else:
        run_headless(env_config)

if __name__ == "__main__":
    main() 