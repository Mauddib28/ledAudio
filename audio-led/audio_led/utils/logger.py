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

# This module provides logging facilities for the Audio LED Visualization System.
# It sets up logging based on configuration settings, with support for different
# log levels, file output, and console output. The logging is designed to be
# non-intrusive in production but detailed enough for debugging.

#--------------------------------------
#       IMPORTS
#--------------------------------------

import os
import sys
import logging
import datetime
from pathlib import Path

#--------------------------------------
#       CONSTANTS
#--------------------------------------

# Log levels mapping
LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

# Default date format for logs
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Default log directory
DEFAULT_LOG_DIR = os.path.join(os.path.expanduser('~'), '.audio_led', 'logs')

#--------------------------------------
#       FUNCTIONS
#--------------------------------------

def setup_logging(level=logging.INFO, log_file=None, log_to_console=True):
    """
    Set up logging with the specified level and outputs.
    
    Args:
        level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): Path to log file. If None, no file logging is done.
        log_to_console (bool): Whether to log to console
        
    Returns:
        logging.Logger: Configured logger
    """
    # Ensure the root logger is configured
    root_logger = logging.getLogger()
    
    # Clear any existing handlers to avoid duplicate logging
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set the log level
    root_logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
    
    # Set up console logging if requested
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Set up file logging if a file path is provided
    if log_file:
        try:
            # Ensure the log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            # Create file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            logging.info(f"Logging to file: {log_file}")
        except Exception as e:
            logging.error(f"Failed to set up file logging: {e}")
    
    return root_logger

def get_log_level(level_name):
    """
    Convert a level name to a logging level constant.
    
    Args:
        level_name (str): Level name (debug, info, warning, error, critical)
        
    Returns:
        int: Corresponding logging level constant
    """
    return LOG_LEVELS.get(level_name.lower(), logging.INFO)

def get_default_log_file():
    """
    Get the default log file path.
    
    Returns:
        str: Default log file path
    """
    # Create a timestamped log file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = DEFAULT_LOG_DIR
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    return os.path.join(log_dir, f'audio_led_{timestamp}.log')

def configure_from_env_config(env_config):
    """
    Configure logging based on environment configuration.
    
    Args:
        env_config (dict): Environment configuration
        
    Returns:
        logging.Logger: Configured logger
    """
    # Extract logging configuration
    system_config = env_config.get('config', {}).get('system', {})
    log_level_name = system_config.get('log_level', 'info')
    log_level = get_log_level(log_level_name)
    
    # Determine if we should log to a file
    log_file = None
    if env_config.get('env_info', {}).get('capabilities', {}).get('file_system', True):
        log_file = get_default_log_file()
    
    # Set up logging
    return setup_logging(log_level, log_file)

def get_module_logger(module_name):
    """
    Get a logger for a specific module.
    
    Args:
        module_name (str): Name of the module
        
    Returns:
        logging.Logger: Logger for the module
    """
    return logging.getLogger(module_name)

# Set up basic logging if this module is run directly
if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    logging.info("Logging module test")
    logging.debug("This is a debug message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    
    # Show where logs will be stored
    print(f"Default log directory: {DEFAULT_LOG_DIR}")
    print(f"Default log file: {get_default_log_file()}") 