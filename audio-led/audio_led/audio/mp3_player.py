#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the Audio-LED project

import os
import io
import logging
import numpy as np
from pydub import AudioSegment

# Initialize logger
logger = logging.getLogger(__name__)

class MP3Player:
    """Simple MP3 player class for Audio-LED project"""
    
    def __init__(self, file_path):
        """Initialize MP3 player with file path
        
        Parameters
        ----------
        file_path : str
            Path to MP3 file
        """
        self.file_path = file_path
        self.position = 0
        self.mp3_data = None
        self.sample_rate = 44100  # Default sample rate
        self.channels = 1  # Default mono
        self.segment = None
        
        # Load the MP3 file
        self.load()
    
    def load(self):
        """Load MP3 file into memory
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading MP3 file: {self.file_path}")
            
            # Load the file with pydub
            self.segment = AudioSegment.from_file(self.file_path, format="mp3")
            
            # Get file properties
            self.sample_rate = self.segment.frame_rate
            self.channels = self.segment.channels
            
            # Convert to mono if stereo
            if self.channels > 1:
                logger.info("Converting stereo MP3 to mono")
                self.segment = self.segment.set_channels(1)
                self.channels = 1
            
            # Convert to numpy array
            self.mp3_data = np.array(self.segment.get_array_of_samples())
            
            logger.info(f"MP3 loaded: {len(self.mp3_data)} samples, {self.sample_rate} Hz")
            return True
            
        except Exception as e:
            logger.error(f"Error loading MP3 file: {str(e)}")
            # If error mentions ffmpeg, add helpful message
            if "ffmpeg" in str(e).lower():
                logger.error("Missing ffmpeg. Install with: 'sudo apt install ffmpeg' or 'sudo pacman -S ffmpeg'")
            return False
    
    def read_chunk(self, chunk_size):
        """Read a chunk of audio data
        
        Parameters
        ----------
        chunk_size : int
            Number of samples to read
            
        Returns
        -------
        numpy.ndarray
            Audio data as numpy array, or None if no more data
        """
        if self.mp3_data is None:
            return None
        
        # Check if we have enough data left
        if self.position + chunk_size <= len(self.mp3_data):
            # Get the chunk
            chunk = self.mp3_data[self.position:self.position + chunk_size]
            self.position += chunk_size
            return chunk
        elif self.position < len(self.mp3_data):
            # Return remaining data
            chunk = self.mp3_data[self.position:]
            self.position = len(self.mp3_data)
            
            # Pad with zeros to reach chunk_size if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
            
            return chunk
        else:
            # End of file
            return None
    
    def rewind(self):
        """Rewind to the beginning of the file"""
        self.position = 0
        
    def close(self):
        """Close the MP3 player and free resources"""
        self.mp3_data = None
        self.segment = None 