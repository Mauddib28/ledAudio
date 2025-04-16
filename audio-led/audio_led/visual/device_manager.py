#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio LED Visualization System - Visual Device Manager
Created on: 2023-08-03
Last modified: 2023-08-03

This module provides utilities for managing and detecting output devices
for the Audio LED Visualization System.
"""

import logging
from .output_handler import OutputHandler

# Set up logging
logger = logging.getLogger(__name__)

class DeviceManager:
    """
    Manages detection and setup of visual output devices.
    
    This class provides methods to detect and list available visual output
    devices for the Audio LED Visualization System.
    """
    
    @staticmethod
    def detect_output_devices():
        """
        Detect available output methods on the current system.
        
        Returns:
            list: A list of available output methods
        """
        return OutputHandler.list_output_methods()
    
    @staticmethod
    def print_output_devices():
        """
        Print a formatted list of available output devices.
        """
        output_methods = DeviceManager.detect_output_devices()
        
        if not output_methods:
            logger.warning("No output methods detected on this system.")
            print("No output methods detected on this system.")
            return
        
        print("\nAvailable output methods:")
        print("-------------------------")
        for i, method in enumerate(output_methods, 1):
            print(f"{i}. {method}")
        print() 