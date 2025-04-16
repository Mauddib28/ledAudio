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
import traceback
from pathlib import Path

# Set up basic logging until proper setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
    else:
        # If GUI mode is requested, explicitly enable display capabilities
        if args.gui:
            if 'capabilities' in env_info:
                env_info['capabilities']['display'] = True
            logging.info("GUI mode enabled, ensuring display capability")
        # Explicitly ensure display capabilities if output is display
        elif cfg['visual']['output_method'] == 'display':
            if 'capabilities' in env_info:
                env_info['capabilities']['display'] = True
            logging.info("Display output mode enabled, ensuring display capability")
    
    # Set colorful visualization mode
    cfg['visual']['color_mode'] = 'colorful'
    
    # Initialize hardware manager but don't start full processing
    logging.info("Initializing hardware manager")
    from audio_led.hardware.device_manager import DeviceManager
    hw_manager = DeviceManager(env_info)
    
    # Return combined configuration
    env_config = {
        'env_info': env_info,
        'config': cfg,
        'args': args,
        'hardware_manager': hw_manager
    }
    
    return env_config

def run_with_gui(env_config):
    """
    Run the application with a graphical user interface.
    
    Args:
        env_config (dict): Environment configuration.
        
    Returns:
        bool: True if execution completed successfully, False otherwise.
    """
    logging.info("Starting with GUI")
    
    try:
        # Import GUI-related modules
        try:
            from audio_led.utils.gui import Application
        except ImportError as e:
            logging.error(f"Failed to import GUI modules: {e}")
            logging.warning("Falling back to headless mode")
            return run_headless(env_config)
        
        # Create and start the application with the environment configuration
        app = Application(env_config)
        
        # Run the application (blocking call)
        app.run()
        
        logging.info("GUI has closed. Cleaning up resources.")
        return True
        
    except Exception as e:
        logging.error(f"Error in GUI mode: {e}")
        return False

# Global flag for graceful exit
should_exit = False

def signal_handler(sig, frame):
    """
    Handle signal interrupts gracefully.
    """
    global should_exit
    should_exit = True
    # Don't log directly from signal handler as it might not be safe
    print("\nReceived interrupt signal, shutting down gracefully...")

# Set up signal handlers early
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_headless(env_config):
    """
    Run the application in headless mode, without a GUI.
    
    Args:
        env_config (dict): Environment configuration.
        
    Returns:
        bool: True if execution completed successfully, False otherwise.
    """
    logging.info("Starting in headless mode")
    
    # Get configuration
    config = env_config.get('config', {})
    
    # Check for essential configuration
    if config.get('audio', {}).get('input_source') is None:
        logging.warning("No input source specified in config")
        input_source = input("Enter input source (file path, 'mic', or 'demo'): ")
        if 'audio' not in config:
            config['audio'] = {}
        config['audio']['input_source'] = input_source
        logging.info(f"User entered input source: {input_source}")
    
    if config.get('visual', {}).get('output_method') is None:
        logging.warning("No output method specified in config")
        output_method = input("Enter output method (led, display, file): ")
        if 'visual' not in config:
            config['visual'] = {}
        config['visual']['output_method'] = output_method
        logging.info(f"User entered output method: {output_method}")
    
    try:
        # Import necessary modules
        from audio_led.audio.input_handler import AudioInputHandler
        from audio_led.audio.processor import AudioProcessor
        from audio_led.visual.rgb_converter import RGBConverter
        from audio_led.visual.output_handler import OutputHandler
        
        # Initialize audio input
        input_source = config['audio']['input_source']
        logging.info(f"Setting up audio input with source: {input_source}")
        audio_input = AudioInputHandler(env_config, device_id=input_source)
        
        if not audio_input.start():
            logging.error("Failed to start audio input")
            return False
        
        # Wait for audio input to initialize with retries
        max_retries = 10
        retry_count = 0
        while not audio_input.is_initialized() and retry_count < max_retries:
            logging.info(f"Waiting for audio input to initialize (attempt {retry_count+1}/{max_retries})...")
            time.sleep(0.5)
            retry_count += 1
        
        if not audio_input.is_initialized():
            logging.error("Failed to initialize audio input after multiple attempts")
            return False
        
        # Initialize audio processor
        logging.info("Initializing audio processor")
        audio_processor = AudioProcessor(env_config)
        audio_processor.audio_input = audio_input
        
        # Initialize RGB converter
        logging.info("Initializing RGB converter")
        rgb_converter = RGBConverter(env_config)
        
        # Initialize output handler using the hardware manager from env_config
        output_method = config['visual']['output_method']
        logging.info(f"Setting up output with method: {output_method}")
        
        # Use the hardware manager that was initialized in setup_environment
        hardware_manager = env_config.get('hardware_manager')
        output_handler = OutputHandler(env_config, hardware_manager)
        output_handler.open()
        
        # Initialize audio playback if the input is a file
        audio_player = None
        if audio_input.input_type == 1:  # INPUT_FILE
            try:
                from audio_led.audio.player import AudioPlayer
                logging.info("Initializing audio playback")
                audio_player = AudioPlayer(env_config)
                audio_player.connect_to_input(audio_input)
                audio_player.start()
                logging.info("Audio playback started")
            except ImportError as e:
                logging.warning(f"Could not import AudioPlayer: {e}")
                logging.warning("Audio playback not available")
            except Exception as e:
                logging.warning(f"Could not start audio playback: {e}")
        
        # Print instructions
        print("\nAudio LED Visualization running in headless mode")
        print(f"Using input source: {input_source}")
        print(f"Output method: {output_method}")
        print("Press Ctrl+C to stop the visualization")
        if 'args' in env_config and hasattr(env_config['args'], 'timeout'):
            print(f"Visualization will automatically stop after {env_config['args'].timeout} seconds\n")
        
        # Main processing loop
        start_time = time.time()
        loop_count = 0
        running = True
        
        logging.info("Starting audio processing loop")
        
        while running and not should_exit:
            try:
                # Check if the timeout has been reached
                if 'args' in env_config and hasattr(env_config['args'], 'timeout') and env_config['args'].timeout > 0:
                    if time.time() - start_time > env_config['args'].timeout:
                        logging.info(f"Timeout reached ({env_config['args'].timeout} seconds)")
                        break
                
                # Process audio data
                audio_data = audio_input.get_audio_chunk()
                
                if audio_data is None or len(audio_data) == 0:
                    logging.warning("No audio data received")
                    time.sleep(0.1)
                    continue
                
                # Process the audio data
                processed_data = audio_processor.process(audio_data)
                
                # Check if we have valid processed data
                if processed_data is None:
                    time.sleep(0.1)
                    continue
                
                # Convert processed data to RGB values
                rgb_values = rgb_converter.convert(processed_data)
                
                # Update output
                if rgb_values is not None:
                    output_handler.update(rgb_values)
                
                # Increment loop count
                loop_count += 1
                
                # Display heartbeat message periodically
                if loop_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {loop_count} audio chunks (running for {int(elapsed)} seconds)")
                
                # Small sleep to prevent CPU overuse
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                logging.info("Keyboard interrupt received. Exiting.")
                running = False
            except Exception as e:
                logging.error(f"Error in audio processing loop: {str(e)}")
                logging.debug(f"Error details: {traceback.format_exc()}")
                time.sleep(0.1)  # Prevent rapid error loops
        
        # Clean up
        logging.info("Shutting down headless mode")
        
        # Stop audio playback
        if audio_player:
            logging.info("Stopping audio playback")
            audio_player.stop()
        
        # Close other resources
        audio_input.close()
        output_handler.close()
        logging.info("Resources cleaned up")
        
        return True
        
    except Exception as e:
        logging.error(f"Error in headless mode: {str(e)}")
        logging.debug(f"Error details: {traceback.format_exc()}")
        return False

def main():
    """
    Main entry point for the Audio LED Visualization System.
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Check if we just need to detect devices and exit
        if args.detect:
            print("\nAudio LED Visualization System - Device Detection\n")
            print("Available audio input devices:")
            try:
                from audio_led.audio.audio_processor import AudioProcessor
                devices = AudioProcessor.list_audio_devices()
                for device in devices:
                    print(f"  - {device}")
            except Exception as e:
                print(f"  Error detecting audio devices: {e}")
            
            print("\nAvailable output devices:")
            try:
                from audio_led.hardware.device_manager import DeviceManager
                manager = DeviceManager()
                manager.detect_devices()
                if manager.output_devices:
                    for device in manager.output_devices:
                        print(f"  - {device}")
                else:
                    print("  No output devices detected")
            except Exception as e:
                print(f"  Error detecting output devices: {e}")
            
            print("\nDetection complete")
            return
        
        # Setup the environment based on arguments
        env_config = setup_environment(args)
        
        # Load configuration
        config = env_config['config']
        
        # Check for early termination
        if should_exit:
            logging.info("Exiting due to interrupt signal")
            return
        
        # Check if GUI mode is requested
        if args.gui:
            logging.info("GUI mode selected")
            # Ensure display capabilities are enabled for GUI
            if 'env_info' not in env_config:
                env_config['env_info'] = {}
            if 'capabilities' not in env_config['env_info']:
                env_config['env_info']['capabilities'] = {}
            env_config['env_info']['capabilities']['display'] = True
            
            # If we're in GUI mode, we don't need to enforce the no_display flag
            if config.get('visual', {}).get('no_display', False):
                logging.info("Overriding no_display setting for GUI mode")
                config['visual']['no_display'] = False
            
            # Run with GUI
            run_with_gui(env_config)
        else:
            # Handle input prompts for headless mode if needed
            if config['audio']['input_source'] is None:
                print("\n=== Audio LED Visualization System ===")
                print("Command-line argument for input source is missing.")
                config['audio']['input_source'] = prompt_for_input()
                logging.info(f"User selected input source: {config['audio']['input_source']}")
            
            if config['visual']['output_method'] is None:
                print("\nCommand-line argument for output method is missing.")
                config['visual']['output_method'] = prompt_for_output()
                logging.info(f"User selected output method: {config['visual']['output_method']}")
                
            # Run in headless mode
            logging.info("Running in headless mode")
            run_headless(env_config)
            
        # Clean exit message
        if should_exit:
            print("Program terminated by user. Cleanup complete.")
        else:
            print("Program finished successfully. Cleanup complete.")
            
    except KeyboardInterrupt:
        print("\nProgram terminated by user interrupt.")
    except Exception as e:
        logging.error(f"Unhandled exception in main: {str(e)}")
        logging.debug(f"Error details: {traceback.format_exc()}")
        print(f"\nAn error occurred: {str(e)}")
    finally:
        # Final cleanup if needed
        logging.info("Exiting main function")

# Main execution check
if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        exit_code = 0
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        traceback.print_exc()
        exit_code = 1
    finally:
        # Ensure a clean exit
        sys.exit(exit_code) 