#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the audio input handler implementation.
"""

import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("test_input_handler")

# Import our module
from audio_led.audio import (
    InputHandlerFactory,
    FileAudioInput,
    MicrophoneAudioInput,
    DEFAULT_SAMPLE_RATE
)

def test_file_input():
    """Test file input handler."""
    logger.info("Testing file input handler")
    
    # Find a test file
    test_dir = Path(__file__).parent.parent / "testWavs"
    test_files = list(test_dir.glob("*.wav"))
    
    if not test_files:
        logger.error("No test files found")
        return
    
    test_file = test_files[0]
    logger.info(f"Using test file: {test_file}")
    
    # Create a file input handler
    handler = InputHandlerFactory.create_handler(
        input_type='file',
        device_id=str(test_file),
        sample_rate=DEFAULT_SAMPLE_RATE,
        channels=2
    )
    
    # Initialize and start
    if not handler.initialize():
        logger.error("Failed to initialize handler")
        return
    
    if not handler.start():
        logger.error("Failed to start handler")
        return
    
    # Process some audio data
    logger.info("Reading audio data...")
    
    # Create arrays to store data for visualization
    chunk_count = 100
    data = []
    volumes = []
    
    # Read chunks
    for i in range(chunk_count):
        chunk = handler.get_audio_chunk()
        if chunk is None:
            if handler.has_audio_finished():
                logger.info("Audio finished")
                break
            time.sleep(0.01)
            continue
            
        # Store data
        if len(data) < 5:  # Only store a few chunks to avoid excessive memory usage
            data.append(chunk)
        volumes.append(handler.get_volume())
        
        # Print progress
        if i % 10 == 0:
            logger.info(f"Processed {i} chunks, volume: {handler.get_volume():.3f}")
    
    # Stop and close
    handler.stop()
    handler.close()
    
    # Plot some data
    plot_data(data, volumes, "File Input")

def test_microphone_input():
    """Test microphone input handler."""
    logger.info("Testing microphone input handler")
    
    # List available devices
    logger.info("Available devices:")
    devices = MicrophoneAudioInput.list_devices()
    for device in devices:
        logger.info(f"  {device['index']}: {device['name']} (channels: {device['channels']}, default: {device['default']})")
    
    # Create a microphone input handler with default device
    handler = InputHandlerFactory.create_handler(
        input_type='microphone',
        sample_rate=DEFAULT_SAMPLE_RATE,
        channels=1
    )
    
    # Initialize and start
    if not handler.initialize():
        logger.error("Failed to initialize handler")
        return
    
    if not handler.start():
        logger.error("Failed to start handler")
        return
    
    # Process some audio data
    logger.info("Reading audio data (5 seconds)...")
    logger.info("Please make some noise into your microphone!")
    
    # Create arrays to store data for visualization
    max_duration = 5  # seconds
    start_time = time.time()
    data = []
    volumes = []
    
    # Read chunks until timeout
    while time.time() - start_time < max_duration:
        chunk = handler.get_audio_chunk()
        if chunk is None:
            time.sleep(0.01)
            continue
            
        # Store data
        if len(data) < 5:  # Only store a few chunks to avoid excessive memory usage
            data.append(chunk)
        volumes.append(handler.get_volume())
        
        # Print volume periodically
        if len(volumes) % 10 == 0:
            logger.info(f"Current volume: {handler.get_volume():.3f}")
    
    # Stop and close
    handler.stop()
    handler.close()
    
    # Plot some data
    plot_data(data, volumes, "Microphone Input")

def plot_data(data, volumes, title):
    """Plot audio data and volumes."""
    logger.info(f"Plotting data for {title}")
    
    try:
        # Create a figure
        plt.figure(figsize=(12, 8))
        
        # Plot waveform of first chunk
        if data:
            plt.subplot(2, 1, 1)
            chunk = data[0]
            plt.plot(chunk)
            plt.title(f"{title} - Waveform")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.grid(True)
        
        # Plot volumes
        plt.subplot(2, 1, 2)
        plt.plot(volumes)
        plt.title(f"{title} - Volume")
        plt.xlabel("Chunk")
        plt.ylabel("Volume")
        plt.grid(True)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(f"{title.lower().replace(' ', '_')}_test.png")
        logger.info(f"Plot saved to {title.lower().replace(' ', '_')}_test.png")
        
    except Exception as e:
        logger.error(f"Error plotting data: {e}")

def main():
    """Run tests."""
    logger.info("Starting input handler tests")
    
    # Test file input
    test_file_input()
    
    # Test microphone input
    try:
        test_microphone_input()
    except Exception as e:
        logger.error(f"Error in microphone test: {e}")
    
    logger.info("Tests completed")

if __name__ == "__main__":
    main() 