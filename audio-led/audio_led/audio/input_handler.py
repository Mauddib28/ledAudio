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

# This module handles audio input for the Audio LED Visualization System.
# It provides a unified interface for capturing audio from different sources:
# - Microphone input
# - WAV file playback
# - MP3 file playback (if pydub is available)
# - Other audio files supported by installed libraries
#
# The audio input is processed in chunks suitable for real-time analysis and visualization.

#--------------------------------------
#       IMPORTS
#--------------------------------------

import os
import sys
import logging
import time
import wave
import numpy as np
from pathlib import Path

# Try to import PyAudio for microphone input and WAV playback
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    logging.warning("PyAudio not available, microphone input will be disabled")

# Try to import pydub for MP3 support
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    logging.warning("Pydub not available, MP3 support will be disabled")

# Try to import scipy for resampling
try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning("SciPy not available, audio resampling will be limited")

# Import AudioProcessor class from processor module
from audio_led.audio.processor import AudioProcessor

# Configure module logger
logger = logging.getLogger(__name__)

#--------------------------------------
#       CONSTANTS
#--------------------------------------

# Audio formats
FORMAT_WAV = "wav"
FORMAT_MP3 = "mp3"
FORMAT_RAW = "raw"
FORMAT_UNKNOWN = "unknown"

# PyAudio format mapping
PYAUDIO_FORMATS = {
    8: pyaudio.paInt8 if HAS_PYAUDIO else None,
    16: pyaudio.paInt16 if HAS_PYAUDIO else None,
    24: pyaudio.paInt24 if HAS_PYAUDIO else None,
    32: pyaudio.paInt32 if HAS_PYAUDIO else None,
    'float32': pyaudio.paFloat32 if HAS_PYAUDIO else None
}

# Default audio parameters
DEFAULT_CHANNELS = 2  # Stereo
DEFAULT_RATE = 44100  # CD quality
DEFAULT_CHUNK = 2048  # Processing chunk size
DEFAULT_FORMAT = 16   # 16-bit audio

# Supported sample rates - most audio interfaces support these
SUPPORTED_SAMPLE_RATES = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000]

#--------------------------------------
#       CLASSES
#--------------------------------------

class AudioInputHandler:
    """
    Handler for audio input from various sources.
    
    This class provides a unified interface for audio input from different sources,
    such as microphones and audio files (WAV, MP3, etc.).
    """
    
    def __init__(self, env_config):
        """
        Initialize the audio input handler.
        
        Args:
            env_config (dict): Environment configuration including audio settings
        """
        self.config = env_config.get('config', {}).get('audio', {})
        self.env_info = env_config.get('env_info', {})
        
        # Initialize audio parameters with defaults and config values
        self.channels = self.config.get('channels', DEFAULT_CHANNELS)
        self.rate = self.config.get('sample_rate', DEFAULT_RATE)
        self.chunk_size = self.config.get('chunk_size', DEFAULT_CHUNK)
        self.format = self.config.get('format', DEFAULT_FORMAT)
        self.play_audio = self.config.get('play_audio', True)
        self.auto_resample = self.config.get('resample', True)  # Enable automatic resampling
        
        # Get the input source from config or prompt user
        self.input_source = self.config.get('input_source', 'auto')
        
        # Initialize state variables
        self.audio = None
        self.stream = None
        self.wf = None
        self.audio_data = None
        self.audio_segment = None
        self.position = 0
        self.frames = []
        self.input_type = None
        self.is_file = False
        self.is_open = False
        self.resample_needed = False
        self.resampled_data = None
        
        # Set up the audio input source
        self._setup_input_source()
    
    def _setup_input_source(self):
        """
        Set up the audio input source based on configuration.
        
        This method determines the input source type and initializes the
        appropriate handlers for that source.
        """
        # Get the input source from config
        source = self.input_source
        
        # Check if we're in GUI mode
        gui_mode = self.config.get('gui_mode', False)
        
        # If we're in GUI mode, don't auto-select input
        if gui_mode:
            logger.info("GUI mode detected, waiting for user to select input source")
            # Don't actually select a source, let the GUI handle it
            return True
        
        # If source is 'auto', try to determine the best one
        if source == 'auto':
            # Check for audio hardware
            if self.env_info.get('capabilities', {}).get('audio_input', False) and HAS_PYAUDIO:
                # Use microphone if available
                logger.info("Auto-selecting microphone input")
                self._setup_microphone_input()
            elif len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
                # Use the file provided as command-line argument
                logger.info(f"Auto-selecting file input from command line: {sys.argv[1]}")
                self._setup_file_input(sys.argv[1])
            else:
                # Look for test WAV files in common locations
                wav_found = False
                
                # Check relative paths
                test_dirs = [
                    Path("testWavs"),
                    Path("../testWavs"),
                    Path("/home/user/Documents/ledAudio/testWavs")
                ]
                
                for test_dir in test_dirs:
                    if test_dir.exists() and test_dir.is_dir():
                        # Find all WAV files
                        wav_files = list(test_dir.glob("*.wav"))
                        if wav_files:
                            # Use the first WAV file found
                            wav_path = str(wav_files[0])
                            logger.info(f"Auto-selecting test WAV file: {wav_path}")
                            if self._setup_file_input(wav_path):
                                wav_found = True
                                break
                
                # If still no file found, prompt the user
                if not wav_found:
                    logger.info("No suitable input automatically detected")
                    self._prompt_for_input()
                
        # If source is 'microphone', use microphone input
        elif source == 'microphone':
            if self.env_info.get('capabilities', {}).get('audio_input', False) and HAS_PYAUDIO:
                logger.info("Using microphone input")
                self._setup_microphone_input()
            else:
                logger.error("Microphone input not available")
                self._prompt_for_input()
                
        # If source is 'file', prompt for a file
        elif source == 'file':
            logger.info("User requested file input")
            self._prompt_for_input()
            
        # If source is a file path, try to use it
        elif os.path.isfile(source):
            logger.info(f"Using file input: {source}")
            self._setup_file_input(source)
            
        # Otherwise, prompt for input
        else:
            logger.warning(f"Unknown input source: {source}")
            self._prompt_for_input()
    
    def _prompt_for_input(self):
        """
        Prompt the user to select an input source.
        
        This method is called when no suitable input source is automatically detected
        or when the user explicitly requests to select a file.
        """
        print("\nSelect audio input source:")
        
        options = []
        
        # Add microphone option if available
        if self.env_info.get('capabilities', {}).get('audio_input', False) and HAS_PYAUDIO:
            options.append(('microphone', "Microphone input"))
            
        # Add test file options from the testWavs directory if it exists
        test_dir = Path("testWavs")
        if test_dir.exists() and test_dir.is_dir():
            for i, file_path in enumerate(test_dir.glob("*.wav")):
                options.append((str(file_path), f"Test WAV: {file_path.name}"))
            
            for i, file_path in enumerate(test_dir.glob("*.mp3")):
                if HAS_PYDUB:
                    options.append((str(file_path), f"Test MP3: {file_path.name}"))
        
        # Display the options
        for i, (_, description) in enumerate(options):
            print(f"  [{i}] {description}")
            
        # Add custom file option
        print(f"  [{len(options)}] Custom file path")
        
        # Get user selection
        while True:
            try:
                choice = int(input("\nEnter selection number: "))
                if 0 <= choice < len(options):
                    selected = options[choice][0]
                    if selected == 'microphone':
                        self._setup_microphone_input()
                    else:
                        self._setup_file_input(selected)
                    break
                elif choice == len(options):
                    # Custom file path
                    file_path = input("Enter audio file path: ")
                    if os.path.isfile(file_path):
                        self._setup_file_input(file_path)
                        break
                    else:
                        print(f"File not found: {file_path}")
                else:
                    print(f"Invalid selection. Please enter a number between 0 and {len(options)}")
            except ValueError:
                print("Please enter a valid number")
    
    def _setup_microphone_input(self):
        """
        Set up microphone input using PyAudio.
        
        This method initializes the PyAudio library for capturing audio from
        the default microphone device.
        """
        if not HAS_PYAUDIO:
            logger.error("PyAudio not available, cannot use microphone input")
            return False
        
        try:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            # Get format
            py_format = PYAUDIO_FORMATS.get(self.format, pyaudio.paInt16)
            
            # Open stream
            self.stream = self.audio.open(
                format=py_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                output=self.play_audio,
                frames_per_buffer=self.chunk_size
            )
            
            self.input_type = "microphone"
            self.is_file = False
            self.is_open = True
            
            # Set up audio processor
            self.audio_processor = AudioProcessor(
                env_config={
                    'config': {
                        'audio': {
                            'sample_rate': self.rate,
                            'channels': self.channels,
                            'chunk_size': self.chunk_size,
                            'format': self.format
                        },
                        'processing': {}
                    }
                }
            )
            
            logger.info(f"Microphone input initialized (rate={self.rate}, channels={self.channels})")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up microphone input: {e}")
            return False
    
    def _setup_file_input(self, file_path):
        """
        Set up audio input from a file.
        
        This method initializes the appropriate library for reading the specified
        audio file based on its format.
        
        Args:
            file_path (str): Path to the audio file
        """
        # Determine file format
        file_ext = os.path.splitext(file_path)[1].lower()[1:]
        
        if file_ext == FORMAT_WAV:
            self._setup_wav_input(file_path)
        elif file_ext == FORMAT_MP3 and HAS_PYDUB:
            self._setup_mp3_input(file_path)
        else:
            logger.error(f"Unsupported audio format: {file_ext}")
            return False
        
        self.is_file = True
        logger.info(f"File input initialized: {file_path}")
        return True
    
    def _setup_wav_input(self, file_path):
        """
        Set up WAV file input.
        
        Args:
            file_path: Path to the WAV file
        
        Returns:
            Success flag
        """
        try:
            # Open the WAV file
            self.wf = wave.open(file_path, 'rb')
            self.channels = self.wf.getnchannels()
            self.original_rate = self.wf.getframerate()
            self.sample_width = self.wf.getsampwidth()
            
            # Check if resampling is needed
            self.resample_needed = self.original_rate != self.rate
            self.using_resampled_audio = False
            self.resampled_audio = None
            self.resampled_position = 0
            
            if self.resample_needed:
                logger.info(f"WAV file sample rate ({self.original_rate}Hz) differs from configured rate ({self.rate}Hz)")
                if self.auto_resample:
                    logger.info("Resampling enabled - will convert audio to match target rate")
                    # Load and resample the entire file
                    if not self._load_and_resample_wav():
                        logger.warning("Resampling failed, will use original sample rate")
                        self.rate = self.original_rate
                        self.resample_needed = False
                else:
                    logger.warning("Resampling disabled - using original sample rate")
                    self.rate = self.original_rate
                    self.resample_needed = False
            else:
                logger.info(f"WAV file sample rate ({self.original_rate}Hz) matches configured rate - no resampling needed")
            
            # Try to open audio stream with target rate
            try:
                # Initialize PyAudio
                self.p = pyaudio.PyAudio()
                
                # Open stream
                self.stream = self.p.open(
                    format=self.p.get_format_from_width(self.sample_width),
                    channels=self.channels,
                    rate=self.rate,
                    output=True
                )
                logger.info(f"Opened audio stream with sample rate {self.rate}Hz")
            except Exception as e:
                logger.error(f"Failed to open audio stream with rate {self.rate}Hz: {e}")
                
                # Try to find closest supported rate
                if not hasattr(self, 'p') or self.p is None:
                    self.p = pyaudio.PyAudio()
                
                # Get device info for default output device
                device_info = self.p.get_default_output_device_info()
                supported_rates = None
                
                # Some devices report supported rates
                if 'supportedSampleRates' in device_info:
                    supported_rates = device_info['supportedSampleRates']
                    logger.info(f"Supported sample rates: {supported_rates}")
                
                if supported_rates:
                    # Find closest supported rate
                    closest_rate = min(supported_rates, key=lambda x: abs(x - self.rate))
                    logger.info(f"Using closest supported sample rate: {closest_rate}Hz")
                    self.rate = closest_rate
                else:
                    # Fallback to common rates
                    common_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000]
                    for rate in sorted(common_rates, key=lambda x: abs(x - self.rate)):
                        try:
                            logger.info(f"Trying sample rate: {rate}Hz")
                            self.stream = self.p.open(
                                format=self.p.get_format_from_width(self.sample_width),
                                channels=self.channels,
                                rate=rate,
                                output=True
                            )
                            self.rate = rate
                            logger.info(f"Successfully opened stream with rate {rate}Hz")
                            break
                        except Exception as e:
                            logger.debug(f"Failed with rate {rate}Hz: {e}")
                    else:
                        # If we get here, all rates failed
                        logger.error("Could not find a working sample rate")
                        return False
                
                # If we need to change resampling based on the new rate
                if self.original_rate != self.rate and not self.using_resampled_audio and self.auto_resample:
                    self.resample_needed = True
                    if not self._load_and_resample_wav():
                        logger.error("Failed to resample for the fallback rate")
                        return False
            
            # Set up audio processor
            self.audio_processor = AudioProcessor(
                env_config={
                    'config': {
                        'audio': {
                            'sample_rate': self.rate,
                            'channels': self.channels,
                            'chunk_size': self.chunk_size,
                            'format': self.sample_width * 8
                        },
                        'processing': {}
                    }
                }
            )
            
            # Set our input type
            self.input_type = "wav_file"
            return True
            
        except Exception as e:
            logger.error(f"Error setting up WAV input: {e}")
            return False
    
    def _load_and_resample_wav(self):
        """Load and resample the entire WAV file into memory"""
        try:
            # Store original file position
            original_position = self.wf.tell()
            self.wf.rewind()
            
            # Read the entire file
            original_audio = self.wf.readframes(self.wf.getnframes())
            
            # Convert to numpy array for resampling
            if self.wf.getsampwidth() == 1:
                dtype = np.uint8
            elif self.wf.getsampwidth() == 2:
                dtype = np.int16
            elif self.wf.getsampwidth() == 3:
                dtype = np.int32
            elif self.wf.getsampwidth() == 4:
                dtype = np.float32
            else:
                raise ValueError(f"Unsupported sample width: {self.wf.getsampwidth()}")
                
            # Reshape audio data based on number of channels
            audio_data = np.frombuffer(original_audio, dtype=dtype)
            
            if self.channels > 1:
                audio_data = audio_data.reshape(-1, self.channels)
            
            logger.info(f"Loaded {len(audio_data)} samples from WAV file")
            
            # Try to use scipy for high-quality resampling if available
            try:
                if HAS_SCIPY:
                    from scipy import signal
                    
                    # Calculate resampling factor
                    resample_factor = self.rate / self.original_rate
                    
                    # For multichannel audio, resample each channel separately
                    if self.channels > 1:
                        resampled_data = np.zeros((int(len(audio_data) * resample_factor), self.channels), dtype=dtype)
                        for ch in range(self.channels):
                            resampled_data[:, ch] = signal.resample(audio_data[:, ch], int(len(audio_data) * resample_factor))
                    else:
                        resampled_data = signal.resample(audio_data, int(len(audio_data) * resample_factor))
                    
                    logger.info(f"Resampled audio using scipy from {self.original_rate}Hz to {self.rate}Hz")
                else:
                    raise ImportError("scipy not available")
            except Exception as e:
                logger.warning(f"Failed to use scipy for resampling: {e}")
                logger.info("Falling back to linear interpolation for resampling")
                
                # Calculate resampling factor
                resample_factor = self.rate / self.original_rate
                
                # For multichannel audio, resample each channel separately
                if self.channels > 1:
                    resampled_length = int(len(audio_data) * resample_factor)
                    resampled_data = np.zeros((resampled_length, self.channels), dtype=dtype)
                    
                    for ch in range(self.channels):
                        # Create interpolation x coordinates (original sample positions)
                        orig_x = np.arange(len(audio_data))
                        # Create interpolation y values (original audio data)
                        orig_y = audio_data[:, ch]
                        # Create x coordinates for resampled data
                        new_x = np.linspace(0, len(audio_data) - 1, resampled_length)
                        # Linear interpolation
                        resampled_data[:, ch] = np.interp(new_x, orig_x, orig_y)
                else:
                    # Create interpolation x coordinates (original sample positions)
                    orig_x = np.arange(len(audio_data))
                    # Create x coordinates for resampled data
                    new_x = np.linspace(0, len(audio_data) - 1, int(len(audio_data) * resample_factor))
                    # Linear interpolation
                    resampled_data = np.interp(new_x, orig_x, audio_data)
                
                logger.info(f"Resampled audio using linear interpolation from {self.original_rate}Hz to {self.rate}Hz")
            
            # Convert back to bytes for playback
            if self.channels > 1:
                self.resampled_audio = resampled_data.astype(dtype).tobytes()
            else:
                self.resampled_audio = resampled_data.astype(dtype).tobytes()
            
            logger.info(f"Resampled audio has {len(self.resampled_audio) // (self.wf.getsampwidth() * self.channels)} frames")
            
            # Reset file pointer to its original position
            self.wf.setpos(original_position)
            
            # We're now using the resampled data for playback
            self.using_resampled_audio = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error during resampling: {e}")
            self.using_resampled_audio = False
            return False
    
    def _setup_mp3_input(self, file_path):
        """
        Set up MP3 file input using pydub.
        
        Args:
            file_path (str): Path to the MP3 file
        """
        if not HAS_PYDUB:
            logger.error("Pydub not available, cannot use MP3 input")
            return False
        
        try:
            # Load the MP3 file
            self.audio_segment = AudioSegment.from_file(file_path, format="mp3")
            
            # Get file properties
            self.channels = self.audio_segment.channels
            self.original_rate = self.audio_segment.frame_rate
            self.format = self.audio_segment.sample_width * 8  # Convert bytes to bits
            
            # Check if we need to resample
            self.resample_needed = False
            target_rate = self.rate
            
            # Use pydub's built-in resampling for MP3
            if self.original_rate != self.rate and self.auto_resample:
                # pydub can handle resampling internally
                logger.info(f"Resampling MP3 from {self.original_rate}Hz to {self.rate}Hz using pydub")
                try:
                    self.audio_segment = self.audio_segment.set_frame_rate(self.rate)
                    logger.info("MP3 resampling completed successfully")
                except Exception as e:
                    logger.error(f"Failed to resample MP3: {e}, using original rate")
                    self.rate = self.original_rate
            else:
                # Use the original rate
                self.rate = self.original_rate
            
            # Convert to raw audio data
            self.audio_data = np.array(self.audio_segment.get_array_of_samples())
            
            if HAS_PYAUDIO and self.play_audio:
                # Initialize PyAudio for playback
                self.audio = pyaudio.PyAudio()
                
                # Get format
                py_format = PYAUDIO_FORMATS.get(self.format, pyaudio.paInt16)
                
                # Open stream for playback
                try:
                    self.stream = self.audio.open(
                        format=py_format,
                        channels=self.channels,
                        rate=self.rate,
                        output=True
                    )
                except Exception as e:
                    logger.error(f"Failed to open audio stream for MP3 playback: {e}")
                    logger.info("Disabling audio playback")
                    self.play_audio = False
            
            self.input_type = "mp3_file"
            self.is_open = True
            
            # Set up audio processor
            self.audio_processor = AudioProcessor(
                env_config={
                    'config': {
                        'audio': {
                            'sample_rate': self.rate,
                            'channels': self.channels,
                            'chunk_size': self.chunk_size,
                            'format': self.format
                        },
                        'processing': {}
                    }
                }
            )
            
            logger.info(f"MP3 file input initialized: {file_path}")
            logger.debug(f"MP3 properties: channels={self.channels}, rate={self.rate}, format={self.format}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up MP3 file input: {e}")
            return False
    
    def get_audio_chunk(self):
        """
        Get the next chunk of audio data.
        
        Returns:
            bytes or numpy.ndarray: Audio data chunk, or None if end of file or error
        """
        if not self.is_open:
            logger.error("Audio input not open")
            return None
        
        try:
            if self.input_type == "microphone":
                # Read from microphone
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                return data
                
            elif self.input_type == "wav_file":
                if self.resample_needed and self.resampled_data is not None:
                    # Read from resampled data
                    samples_per_chunk = self.chunk_size
                    start = self.position
                    end = min(start + samples_per_chunk, len(self.resampled_data))
                    
                    # If at the end of the data, return None
                    if start >= len(self.resampled_data):
                        return None
                    
                    # Get the chunk of resampled data
                    chunk = self.resampled_data[start:end]
                    
                    # Update position
                    self.position = end
                    
                    # Convert to bytes for playback if needed
                    if self.play_audio and self.stream:
                        try:
                            self.stream.write(chunk.tobytes())
                        except Exception as e:
                            logger.error(f"Error during audio playback: {e}")
                    
                    return chunk
                else:
                    # Read from WAV file directly
                    data = self.wf.readframes(self.chunk_size)
                    
                    # If at the end of the file, return None
                    if len(data) == 0:
                        return None
                    
                    # Play the audio if requested
                    if self.stream and self.play_audio:
                        try:
                            self.stream.write(data)
                        except Exception as e:
                            logger.error(f"Error during audio playback: {e}")
                    
                    return data
                
            elif self.input_type == "mp3_file":
                # Calculate the number of samples in a chunk
                samples_per_chunk = self.chunk_size * self.channels
                
                # Calculate start and end positions
                start = self.position
                end = min(start + samples_per_chunk, len(self.audio_data))
                
                # If at the end of the file, return None
                if start >= len(self.audio_data):
                    return None
                
                # Get the chunk of audio data
                data = self.audio_data[start:end]
                
                # Update position
                self.position = end
                
                # Play the audio if requested
                if self.stream and self.play_audio:
                    try:
                        self.stream.write(data.tobytes())
                    except Exception as e:
                        logger.error(f"Error during audio playback: {e}")
                
                return data
                
            else:
                logger.error(f"Unsupported input type: {self.input_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting audio chunk: {e}")
            return None
    
    def close(self):
        """
        Close the audio input and clean up resources.
        """
        if not self.is_open:
            return
        
        logger.info("Closing audio input")
        
        try:
            # Close stream
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            # Close PyAudio
            if self.audio:
                self.audio.terminate()
            
            # Close WAV file
            if self.wf:
                self.wf.close()
            
            self.is_open = False
            
        except Exception as e:
            logger.error(f"Error closing audio input: {e}")

# Test the module if run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple environment configuration
    env_config = {
        'config': {
            'audio': {
                'input_source': 'auto',
                'play_audio': True
            }
        },
        'env_info': {
            'capabilities': {
                'audio_input': True,
                'audio_output': True
            }
        }
    }
    
    # Initialize the audio input handler
    audio_input = AudioInputHandler(env_config)
    
    # Print audio input properties
    print(f"\nAudio Input Properties:")
    print(f"  Type: {audio_input.input_type}")
    print(f"  Channels: {audio_input.channels}")
    print(f"  Sample Rate: {audio_input.rate}")
    print(f"  Format: {audio_input.format}-bit")
    print(f"  Chunk Size: {audio_input.chunk_size}")
    
    # Read and process a few chunks
    print("\nReading 5 chunks of audio data...")
    for i in range(5):
        data = audio_input.get_audio_chunk()
        if data is None:
            print("  End of audio input")
            break
        
        print(f"  Chunk {i+1}: {len(data)} bytes")
        time.sleep(0.1)
    
    # Close the audio input
    audio_input.close() 