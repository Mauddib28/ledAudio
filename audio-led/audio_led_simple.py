#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio LED Visualization - Simple Player

A simplified version of the audio LED visualization system that focuses
on playing audio files with a GUI that displays the filename.
"""

import os
import sys
import argparse
import time
import threading
import logging
import numpy as np
import pygame
from pygame import gfxdraw
import traceback
from collections import deque
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("audio_led_simple")

# Try to import audio libraries with graceful fallbacks
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    logger.warning("PyAudio not available - install with: pip install pyaudio")

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    logger.warning("PyDub not available - install with: pip install pydub")

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    logger.warning("Soundfile not available - install with: pip install soundfile")

# Fallbacks for missing libraries
if not HAS_SOUNDFILE:
    try:
        from scipy.io import wavfile
        HAS_SCIPY_WAVFILE = True
    except ImportError:
        HAS_SCIPY_WAVFILE = False
        logger.warning("Scipy wavfile not available - install with: pip install scipy")

# Constants
SAMPLE_RATE = 44100
CHANNELS = 2
CHUNK_SIZE = 2048  # Reverted from 4096 to reduce stuttering
BUFFER_SIZE = 8192  # Larger buffer for smoother playback
MAX_BUFFER_SIZE = 30  # Keep increased buffer capacity
FFT_SIZE = 2048  # Size for FFT calculation (power of 2)
MAX_FPS = 60     # Cap the frame rate to reduce CPU usage

# Revised processing frequency constants
FFT_PROCESS_EVERY = 5  # Reduced from 10 to improve visualization responsiveness
LOG_CALLBACK_EVERY = 200  # Keep this value to reduce log overhead
LOG_AUDIO_CHUNK_EVERY = 300  # Keep this value to reduce log overhead

# Add a constant for lock timeout
BUFFER_LOCK_TIMEOUT = 0.1  # Increased from 0.05 to reduce frequent silence periods

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)

class AudioPlayer:
    """Audio player with visualization."""
    def __init__(self):
        """Initialize the audio player."""
        self.audio_file = None
        self.file_name = None
        self.stream = None
        self.p = None
        self.running = False
        self.finished = False
        self.paused = False
        self.audio_device_available = False
        
        # Audio properties
        self.sample_rate = 44100
        self.channels = 2
        self.chunk_size = CHUNK_SIZE  # CHANGE 4: Use the increased chunk size constant
        self.frames_per_buffer = 1024  # CHANGE 5: Smaller frames per buffer for callback, larger chunks for buffering
        
        # CHANGE 6: Buffer for audio data with increased size
        self.buffer = deque(maxlen=MAX_BUFFER_SIZE)
        self.buffer_lock = threading.RLock()
        
        # Visualization data
        self.fft_data = None
        self.current_rgb = (0, 0, 0)
        self.frame_count = 0
        self.fft_update_rate = FFT_PROCESS_EVERY  # CHANGE 7: Reduced FFT processing frequency
        
        # Stats for monitoring
        self.underrun_count = 0
        self.last_underrun_log = 0  # To avoid log spamming
        
        # Duration and position
        self.duration = 0
        self.position = 0
        self.last_position_update = 0
        
        # Ensure dummy_mode is initialized to False - no more fallback mode
        self.dummy_mode = False
        self.visualization_timer = None
        
        # CHANGE 8: Add flag for threaded buffer reloading
        self._reload_in_progress = False

    def load_file(self, file_path):
        """Load an audio file."""
        try:
            if self.running:
                self.stop()
                
            # Clear any existing data
            with self.buffer_lock:
                self.buffer.clear()
                
            self.audio_file = None
            self.file_name = None
            self.finished = False
            self.duration = 0
            self.position = 0
            
            # Get file information
            if not os.path.isfile(file_path):
                logger.error(f"File not found: {file_path}")
                return False
                
            # Check file extension
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in ['.mp3', '.wav', '.ogg', '.flac']:
                logger.error(f"Unsupported file format: {ext}")
                return False
                
            try:
                # Attempt to load the file with detailed error logging
                audio_data = None
                
                # Try to load the file using soundfile if available
                if HAS_SOUNDFILE:
                    try:
                        logger.debug(f"Attempting to load with SoundFile: {file_path}")
                        audio_data, self.sample_rate = sf.read(file_path)
                        logger.debug(f"SoundFile load successful: {len(audio_data)} frames, {self.sample_rate} Hz")
                    except Exception as sf_error:
                        logger.warning(f"SoundFile load failed: {sf_error}, trying alternative")
                
                # Fallback to scipy for WAV files only
                if audio_data is None and HAS_SCIPY_WAVFILE and ext == '.wav':
                    try:
                        logger.debug(f"Attempting to load with scipy.io.wavfile: {file_path}")
                        self.sample_rate, audio_data = wavfile.read(file_path)
                        logger.debug(f"Scipy wavfile load successful: {len(audio_data)} frames, {self.sample_rate} Hz")
                    except Exception as scipy_error:
                        logger.warning(f"Scipy wavfile load failed: {scipy_error}, trying alternative")
                
                # Fallback to pydub for other formats
                if audio_data is None and HAS_PYDUB:
                    try:
                        logger.debug(f"Attempting to load with PyDub: {file_path}")
                        audio_segment = AudioSegment.from_file(file_path)
                        self.sample_rate = audio_segment.frame_rate
                        samples = np.array(audio_segment.get_array_of_samples())
                        
                        # Convert to proper numpy array and reshape to match channels
                        if audio_segment.channels == 2:
                            samples = samples.reshape((-1, 2))
                        audio_data = samples.astype(np.float32) / 32768.0  # Convert to -1.0 to 1.0 range
                        logger.debug(f"PyDub load successful: {len(audio_data)} frames, {self.sample_rate} Hz")
                    except Exception as pydub_error:
                        logger.warning(f"PyDub load failed: {pydub_error}")
                
                if audio_data is None:
                    logger.error("All audio loading methods failed")
                    return False
                
                # Get file duration with a safety margin
                file_duration = len(audio_data) / self.sample_rate
                # Add a small buffer to prevent premature ending (0.5 seconds)
                self.duration = file_duration + 0.5  
                logger.debug(f"File duration: {file_duration:.2f}s, adjusted duration: {self.duration:.2f}s")
                
                # Convert to float32 and normalize to -1.0 to 1.0 range
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                    
                # Scale to -1.0 to 1.0 if needed
                max_abs = np.max(np.abs(audio_data))
                if max_abs > 1.0:
                    logger.debug(f"Normalizing audio data with max value: {max_abs}")
                    audio_data = audio_data / max_abs
                
                # Handle mono audio
                if len(audio_data.shape) == 1:
                    # Convert mono to stereo by duplicating the channel
                    logger.debug("Converting mono audio to stereo")
                    audio_data = np.column_stack((audio_data, audio_data))
                    self.channels = 2
                else:
                    self.channels = min(2, audio_data.shape[1])  # Limit to 2 channels
                    if audio_data.shape[1] > 2:
                        logger.warning(f"Audio has {audio_data.shape[1]} channels, using first 2 only")
                        audio_data = audio_data[:, :2]
                
                # Check for any NaN or infinite values
                if not np.all(np.isfinite(audio_data)):
                    logger.warning("Non-finite values detected in loaded audio - fixing")
                    audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Split into chunks and add to buffer
                chunk_samples = self.chunk_size
                total_chunks = (len(audio_data) + chunk_samples - 1) // chunk_samples
                logger.debug(f"Splitting audio into {total_chunks} chunks of {chunk_samples} samples each")
                
                with self.buffer_lock:
                    for i in range(0, len(audio_data), chunk_samples):
                        chunk = audio_data[i:i+chunk_samples]
                        # Pad the last chunk if necessary
                        if len(chunk) < chunk_samples:
                            padding = np.zeros((chunk_samples - len(chunk), self.channels), dtype=np.float32)
                            chunk = np.vstack((chunk, padding))
                        self.buffer.append(chunk)
                
                self.audio_file = audio_data
                self.file_name = os.path.basename(file_path)
                buffer_size = len(self.buffer)
                logger.info(f"Loaded file: {self.file_name}, {self.sample_rate} Hz, {self.channels} channels, {buffer_size} chunks")
                
                # Initialize FFT data with zeros
                self.fft_data = np.zeros(FFT_SIZE // 2, dtype=np.float32)
                self.current_rgb = (0, 0, 0)
                
                return True
                
            except Exception as e:
                logger.error(f"Error loading audio file: {e}")
                traceback.print_exc()
                return False
                
        except Exception as e:
            logger.error(f"Error in load_file: {e}")
            traceback.print_exc()
            return False

    def start(self):
        """Start audio playback."""
        if self.audio_file is None:
            logger.error("No audio file loaded")
            return False
            
        if self.running:
            logger.info("Already playing")
            return True
        
        # Start with real audio output only - no fallback to dummy mode
        success = self._start_with_audio()
        return success
    
    def _start_with_audio(self):
        """Start with real audio output."""
        try:
            # Reset playback variables first to ensure clean state
            self.position = 0
            self.last_position_update = time.time()
            self.frame_count = 0
            self.finished = False
            self.underrun_count = 0
            
            # Reset any previous flag
            if hasattr(self, '_logged_inactive_warning'):
                del self._logged_inactive_warning
            
            # Reset buffer reload flag
            self._reload_in_progress = False
            
            # Improved sample rate handling
            original_sample_rate = self.sample_rate
            if self.sample_rate not in (44100, 48000):
                logger.warning(f"Non-standard sample rate {self.sample_rate}Hz, converting to 44100Hz")
                # Store original sample rate for proper position calculation
                self._original_sample_rate = self.sample_rate
                self.sample_rate = 44100
                
                # If we have audio data and the sample rate needs conversion
                if self.audio_file is not None and hasattr(self, '_original_sample_rate'):
                    logger.info(f"Resampling audio from {self._original_sample_rate}Hz to {self.sample_rate}Hz")
                    try:
                        # Simple linear interpolation resampling (better to use a library like librosa if available)
                        ratio = self.sample_rate / self._original_sample_rate
                        new_length = int(len(self.audio_file) * ratio)
                        
                        # Create resampled array
                        channels = self.audio_file.shape[1] if len(self.audio_file.shape) > 1 else 1
                        resampled = np.zeros((new_length, channels), dtype=np.float32)
                        
                        # Simple resampling - for each new sample, find the proportional position in original array
                        for i in range(new_length):
                            orig_idx = int(i / ratio)
                            if orig_idx < len(self.audio_file):
                                resampled[i] = self.audio_file[orig_idx]
                        
                        # Replace audio file with resampled version
                        self.audio_file = resampled
                        
                        # Update duration to account for new sample rate
                        # Original duration is maintained (same audio length in seconds)
                        logger.info(f"Resampling complete: {len(self.audio_file)} frames")
                        
                        # Clear and reload buffer with resampled audio
                        with self.buffer_lock:
                            self.buffer.clear()
                            self._load_buffer_from_audio_file()
                    except Exception as e:
                        logger.error(f"Error resampling audio: {e}")
                        # Continue with original audio data if resampling fails
                        
            # Handle ALSA errors by redirecting stderr temporarily
            import io
            import sys
            alsa_errors = io.StringIO()
            original_stderr = sys.stderr
            sys.stderr = alsa_errors
            
            # Initialize PyAudio
            logger.debug("Initializing PyAudio...")
            try:
                self.p = pyaudio.PyAudio()
                # Log any ALSA errors that occurred
                alsa_error_text = alsa_errors.getvalue()
                if alsa_error_text:
                    logger.debug(f"ALSA messages during initialization: {alsa_error_text.strip()}")
            finally:
                sys.stderr = original_stderr
                
            # Log available devices for debugging
            info = self.p.get_host_api_info_by_index(0)
            device_count = info.get('deviceCount', 0)
            logger.debug(f"Available audio devices: {device_count}")
            for i in range(device_count):
                device_info = self.p.get_device_info_by_index(i)
                logger.debug(f"Device {i}: {device_info.get('name')} (out={device_info.get('maxOutputChannels')}, in={device_info.get('maxInputChannels')})")
            
            # Find default output device
            default_output_device = self.p.get_default_output_device_info()
            logger.debug(f"Default output device: {default_output_device.get('name')}")
            
            # Verify buffer has data and reload if needed
            with self.buffer_lock:
                buffer_size = len(self.buffer)
                logger.debug(f"Buffer status: {buffer_size} chunks available")
                if buffer_size == 0:
                    logger.debug("Reloading buffer before playback")
                    self._load_buffer_from_audio_file()  # Reload buffer
                    buffer_size = len(self.buffer)
                    if buffer_size == 0:
                        logger.error("Failed to load audio buffer")
                        self._cleanup()
                        return False
                    logger.debug(f"After reload: {buffer_size} chunks available")
            
            # Reset stream data in case of previous errors
            self.stream = None
            
            # Basic stream setup - use a simpler approach first
            logger.debug(f"Opening audio stream with: {self.channels} channels, {self.sample_rate} Hz rate")
            
            # Redirect stderr again for stream opening
            sys.stderr = alsa_errors
            alsa_errors.truncate(0)
            alsa_errors.seek(0)
            
            try:
                # Try a simpler, direct approach first with blocking mode
                logger.debug("Attempting standard configuration")
                try:
                    # Create a smaller frames_per_buffer for more reliable playback
                    adjusted_frames = 1024
                    self.stream = self.p.open(
                        format=pyaudio.paFloat32,
                        channels=self.channels,
                        rate=self.sample_rate,
                        output=True,
                        output_device_index=default_output_device.get('index'),
                        frames_per_buffer=adjusted_frames
                    )
                    
                    # Test the stream by writing a small amount of silence
                    test_data = np.zeros(adjusted_frames * self.channels, dtype=np.float32)
                    self.stream.write(test_data.tobytes())
                    
                    # If we got here, then the stream is working - switch to callback mode
                    self.stream.stop_stream()
                    self.stream.close()
                    
                    logger.debug("Opening callback-based stream after successful test")
                    self.stream = self.p.open(
                        format=pyaudio.paFloat32,
                        channels=self.channels,
                        rate=self.sample_rate,
                        output=True,
                        output_device_index=default_output_device.get('index'),
                        frames_per_buffer=adjusted_frames,
                        stream_callback=self._pyaudio_callback
                    )
                except Exception as first_error:
                    logger.warning(f"Error opening audio stream: {first_error}")
                    # Try with a different configuration
                    try:
                        logger.debug("Trying simpler configuration")
                        self.stream = self.p.open(
                            format=pyaudio.paFloat32,
                            channels=1,  # Try mono
                            rate=44100,  # Standard rate
                            output=True,
                            frames_per_buffer=1024
                        )
                        
                        # Test the stream by writing a small amount of silence
                        test_data = np.zeros(1024, dtype=np.float32)
                        self.stream.write(test_data.tobytes())
                        
                        # If we got here, switch to mono and use callback mode
                        self.stream.stop_stream()
                        self.stream.close()
                        
                        # Update parameters to match the working configuration
                        self.channels = 1
                        self.sample_rate = 44100
                        self.frames_per_buffer = 1024
                        
                        logger.debug("Opening callback-based mono stream")
                        self.stream = self.p.open(
                            format=pyaudio.paFloat32,
                            channels=1,
                            rate=44100,
                            output=True,
                            frames_per_buffer=1024,
                            stream_callback=self._pyaudio_callback
                        )
                    except Exception as retry_error:
                        logger.error(f"All stream configurations failed: {retry_error}")
                        raise
            finally:
                # Restore stderr and log any ALSA errors
                sys.stderr = original_stderr
                alsa_error_text = alsa_errors.getvalue()
                if alsa_error_text:
                    if "unable to open slave" in alsa_error_text:
                        logger.warning("ALSA error: Unable to open slave device. This may be due to another application using the audio device.")
                    else:
                        logger.debug(f"ALSA messages during stream opening: {alsa_error_text.strip()}")
            
            if not self.stream:
                logger.error("Failed to create PyAudio stream")
                self._cleanup()
                return False
                
            # Reset playback position
            self.position = 0
            self.last_position_update = time.time()
            self.frame_count = 0
            self.finished = False
            self.running = True
            self.underrun_count = 0
            self.dummy_mode = False
            
            # Start the stream
            logger.debug("Starting audio stream...")
            
            # Check stream state before starting
            if not self.stream.is_active():
                self.stream.start_stream()
                
                # Verify the stream started successfully
                if not self.stream.is_active():
                    logger.error("Stream failed to start - inactive immediately after start_stream()")
                    self._cleanup()
                    return False
                    
                logger.info(f"Playback started: {self.file_name}")
                self.audio_device_available = True
            else:
                logger.info(f"Stream already active for {self.file_name}")
                self.audio_device_available = True
            
            return True
        except Exception as e:
            logger.error(f"Error starting audio playback: {e}")
            traceback.print_exc()
            self._cleanup()
            return False

    def _load_buffer_from_audio_file(self):
        """Reload buffer from audio file if it's empty."""
        logger.debug("Loading audio buffer from file")
        if self.audio_file is None:
            logger.error("Cannot load buffer - no audio file loaded")
            return
            
        # Calculate total frames in audio file
        total_frames = len(self.audio_file)
        
        # Calculate current frame position based on playback position
        current_frame = int(self.position * self.sample_rate)
        current_frame = max(0, min(current_frame, total_frames - 1))
        
        # If we're near the end, don't bother reloading
        if current_frame >= total_frames - self.chunk_size:
            logger.debug(f"Near end of file ({current_frame}/{total_frames}), no need to reload buffer")
            return
            
        # CHANGE 9: Only clear and reload if buffer is very low or empty
        buffer_size = len(self.buffer)
        if buffer_size > 5:  # Still have some chunks
            logger.debug(f"Buffer still has {buffer_size} chunks, skipping reload")
            return
            
        # Clear any existing data
        self.buffer.clear()
        
        # Calculate how many chunks to load from current position
        chunk_samples = self.chunk_size
        remaining_frames = total_frames - current_frame
        
        # CHANGE: Load as much as the buffer can hold - preload aggressively
        chunks_to_load = min(self.buffer.maxlen, (remaining_frames + chunk_samples - 1) // chunk_samples)
        
        logger.debug(f"Reloading buffer at position {self.position:.2f}s (frame {current_frame}/{total_frames}), loading {chunks_to_load} chunks")
        
        # Load chunks starting from the current frame position
        audio_data = self.audio_file
        loaded_chunks = 0
        
        for i in range(current_frame, total_frames, chunk_samples):
            if loaded_chunks >= self.buffer.maxlen:
                break
                
            end_idx = min(i + chunk_samples, total_frames)
            chunk = audio_data[i:end_idx]
            
            # Pad the last chunk if necessary
            if len(chunk) < chunk_samples:
                padding = np.zeros((chunk_samples - len(chunk), self.channels), dtype=np.float32)
                chunk = np.vstack((chunk, padding))
            
            # Add the chunk to the buffer
            self.buffer.append(chunk)
            loaded_chunks += 1
        
        buffer_size = len(self.buffer)
        buffer_duration = buffer_size * chunk_samples / self.sample_rate
        logger.info(f"Loaded buffer with {buffer_size} chunks ({buffer_duration:.1f}s of audio) starting at position {self.position:.2f}s")
        
        # Add a small amount to the duration to account for potential rounding errors
        if self.duration < buffer_duration:
            logger.debug(f"Adjusting duration from {self.duration}s to {buffer_duration}s")
            self.duration = buffer_duration

    # Improve threading coordination for buffer reloading
    def _threaded_buffer_reload(self):
        """Reload the buffer in a separate thread to avoid blocking audio."""
        try:
            # Don't reload if another reload is already in progress
            if getattr(self, '_reload_running', False):
                logger.debug("Skipping buffer reload - another reload already in progress")
                self._reload_in_progress = False
                return
                
            # Set flag to indicate active reload operation
            self._reload_running = True
            
            # Actual buffer reload
            self._load_buffer_from_audio_file()
            
            # Clear reload running flag
            self._reload_running = False
        except Exception as e:
            logger.error(f"Error in threaded buffer reload: {e}")
        finally:
            # Always clear the in-progress flag when done
            self._reload_in_progress = False

    def stop(self):
        """Stop audio playback."""
        if not self.running:
            logger.debug("Stop requested but player not running")
            return True

        # Save current state
        was_playing = self.is_playing()
        
        # Set flags first to stop callback processing
        logger.debug("Setting running=False to stop playback")
        self.running = False
        
        try:
            # Special handling for different ending conditions
            if was_playing:
                logger.info(f"Stopping active playback of {self.file_name}")
            elif self.finished:
                logger.info(f"Playback already finished for {self.file_name}")
            else:
                logger.info(f"Stopping inactive playback of {self.file_name}")
                
            # Call cleanup to properly release resources
            self._cleanup()
        except Exception as e:
            logger.error(f"Error in stop method: {e}")
            traceback.print_exc()
        
        logger.info("Playback stopped")
        return True

    def _cleanup(self):
        """Clean up resources."""
        try:
            # Stop audio stream with proper error handling
            if hasattr(self, 'stream') and self.stream:
                try:
                    if self.stream.is_active():
                        logger.debug("Stopping active stream")
                        self.stream.stop_stream()
                except Exception as stream_error:
                    logger.error(f"Error stopping stream: {stream_error}")
                
                try:
                    logger.debug("Closing stream")
                    self.stream.close()
                except Exception as close_error:
                    logger.error(f"Error closing stream: {close_error}")
                    
                self.stream = None
                
            # Terminate PyAudio with error handling
            if hasattr(self, 'p') and self.p:
                try:
                    logger.debug("Terminating PyAudio")
                    self.p.terminate()
                except Exception as pa_error:
                    logger.error(f"Error terminating PyAudio: {pa_error}")
                
                self.p = None
            
            # Stop visualization thread if it exists
            self.dummy_mode = False
            if hasattr(self, 'visualization_timer') and self.visualization_timer and self.visualization_timer.is_alive():
                try:
                    logger.debug("Joining visualization timer thread")
                    self.visualization_timer.join(timeout=1.0)
                except Exception as thread_error:
                    logger.error(f"Error joining thread: {thread_error}")
                
                self.visualization_timer = None
                
            # Clear buffers to free memory
            with self.buffer_lock:
                self.buffer.clear()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            traceback.print_exc()

    def is_playing(self):
        """Check if audio is playing."""
        return self.running and not self.finished and not self.paused

    def is_finished(self):
        """Check if playback has finished."""
        if not self.running:
            return self.finished
        
        # Consider finished if explicitly set, or if we've reached/exceeded the file duration
        # Also check if the stream is active (if it exists)
        if hasattr(self, 'stream') and self.stream:
            is_active = self.stream.is_active()
            if not is_active:
                logger.debug(f"is_finished: Stream is no longer active, setting finished=True")
                self.finished = True
            
        if self.finished:
            logger.debug(f"is_finished: Returning True, position={self.position:.2f}s, duration={self.duration:.2f}s")
        
        return self.finished or self.position >= self.duration

    def get_progress(self):
        """Get the current playback progress as a float between 0 and 1."""
        if self.audio_file is None or self.duration <= 0:
            return 0.0
            
        # Only update the position every 100ms to reduce unnecessary calculations
        current_time = time.time()
        if self.running and current_time - self.last_position_update >= 0.1:
            # Update position based on time elapsed
            elapsed = current_time - self.last_position_update
            self.last_position_update = current_time
            
            # Only advance position if we're still playing
            if self.stream and self.stream.is_active():
                self.position += elapsed
            else:
                # CHANGE 14: Improved stream inactive handling
                # Log warning only once
                if not getattr(self, '_logged_inactive_warning', False):
                    logger.warning(f"Stream inactive during playback at position {self.position:.2f}s")
                    self._logged_inactive_warning = True
                    
                    # Improved stream restart logic
                    remaining_duration = self.duration - self.position
                    if remaining_duration > 0.5:  # If significant audio remains (reduced from 1.0)
                        logger.info(f"Attempting to restart stream with {remaining_duration:.2f}s remaining")
                        try:
                            # CHANGE: Create a new stream instead of trying to restart the inactive one
                            if hasattr(self, 'stream') and self.stream:
                                try:
                                    self.stream.close()  # Close the inactive stream
                                except Exception as close_error:
                                    logger.error(f"Error closing inactive stream: {close_error}")
                            
                            # Re-initialize with safe parameters
                            try:
                                self.stream = self.p.open(
                                    format=pyaudio.paFloat32,
                                    channels=1,  # Use mono as safer option
                                    rate=44100,  # Standard rate
                                    output=True,
                                    frames_per_buffer=1024,
                                    stream_callback=self._pyaudio_callback
                                )
                                logger.info("Created new stream after inactivity")
                                # Reset the warning flag to allow future warnings if needed
                                self._logged_inactive_warning = False
                            except Exception as e:
                                logger.error(f"Failed to create new stream: {e}")
                                self.finished = True
                        except Exception as e:
                            logger.error(f"Failed to restart stream: {e}")
                            self.finished = True
                            
        # CHANGE: Add back check for end of file
        if self.position >= self.duration:
            if not self.finished:
                logger.info(f"Playback complete for {self.file_name}")
                self.finished = True
                
        # Return a value between 0 and 1
        return min(1.0, self.position / self.duration)

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for streaming audio data."""
        # Check and log stream status
        if status:
            status_text = str(status)
            logger.warning(f"PyAudio status: {status_text}")
            if "underflow" in status_text.lower():
                self.underrun_count += 1
                logger.warning(f"Buffer underrun detected ({self.underrun_count})")
        
        # Reduce logging frequency to decrease overhead
        if self.frame_count % LOG_CALLBACK_EVERY == 0:
            logger.debug(f"Callback frame {self.frame_count}: position={self.position:.2f}s/{self.duration:.2f}s, buffer={len(self.buffer)}/{self.buffer.maxlen}")
            
        # Check if playback should be stopped
        if not self.running:
            logger.debug("Playback stopped - not running")
            return (np.zeros(frame_count * self.channels, dtype=np.float32).tobytes(), pyaudio.paComplete)
        
        # Check if we've reached the end of the file
        if self.position >= self.duration:
            self.finished = True
            logger.debug(f"Playback complete - reached end of duration: {self.position:.2f}s >= {self.duration:.2f}s")
            return (np.zeros(frame_count * self.channels, dtype=np.float32).tobytes(), pyaudio.paComplete)
        
        # Create default output in case of error
        output_data = np.zeros((frame_count, self.channels), dtype=np.float32)
        
        # Try to acquire lock with adjusted timeout
        lock_acquired = self.buffer_lock.acquire(timeout=BUFFER_LOCK_TIMEOUT)
        try:
            if not lock_acquired:
                logger.warning("Buffer lock acquisition timed out in callback")
                # Return silence but continue playback
                return (output_data.flatten().tobytes(), pyaudio.paContinue)
            
            # Enhanced buffering - trigger reload earlier but less frequently
            # Only start a new thread if buffer is low and no reload is already in progress
            if len(self.buffer) < 5 and not self._reload_in_progress and not getattr(self, '_reload_running', False):
                # Run reload in a separate thread to avoid blocking audio
                self._reload_in_progress = True
                thread = threading.Thread(target=self._threaded_buffer_reload)
                thread.daemon = True  # Make sure thread doesn't block program exit
                thread.start()
            
            # Check if buffer is empty
            if not self.buffer:
                logger.warning(f"Buffer empty at position {self.position:.2f}s/{self.duration:.2f}s")
                
                # Only return complete if definitely at the end
                if self.position >= self.duration * 0.95:
                    self.finished = True
                    logger.debug("Playback complete - buffer empty and near end")
                    return (output_data.flatten().tobytes(), pyaudio.paComplete)
                else:
                    # Return silence but keep playing - prevents premature end
                    logger.warning(f"Buffer underrun during playback at position {self.position:.2f}s")
                    return (output_data.flatten().tobytes(), pyaudio.paContinue)
            
            # Get data from buffer
            chunk = self.buffer.popleft()
                        
            # Process the chunk for visualization - improved scheduling
            self.frame_count += 1
            
            # More efficient FFT update scheduling
            if self.frame_count % self.fft_update_rate == 0:
                try:
                    # Make a copy of chunk to prevent threading issues
                    chunk_copy = np.copy(chunk)
                    # Process FFT in the main thread but only every N frames
                    self._update_fft(chunk_copy)
                except Exception as e:
                    logger.error(f"Error updating FFT: {e}")
            
            # Make sure chunk has the right shape and size
            if len(chunk.shape) != 2:
                # Reshape single-dimensional data
                logger.warning(f"Chunk has unexpected shape: {chunk.shape}, reshaping")
                chunk = chunk.reshape((-1, self.channels))
            
            if chunk.shape[0] > frame_count:
                # Trim if too large
                chunk = chunk[:frame_count]
            elif chunk.shape[0] < frame_count:
                # Pad with zeros if too small
                padding = np.zeros((frame_count - chunk.shape[0], self.channels), dtype=np.float32)
                chunk = np.vstack((chunk, padding))
                
            # Check and fix for any channel count issues
            if chunk.shape[1] != self.channels:
                # More detailed logging to diagnose any channel mismatch issues
                logger.warning(f"Channel count mismatch: chunk has {chunk.shape[1]}, expected {self.channels}")
                if chunk.shape[1] < self.channels:
                    # Duplicate channels if needed
                    chunk = np.column_stack([chunk] * (self.channels // chunk.shape[1] + 1))[:, :self.channels]
                else:
                    # Use only the requested number of channels
                    chunk = chunk[:, :self.channels]
            
            # Check for NaN or Inf values
            if not np.all(np.isfinite(chunk)):
                logger.warning("Non-finite values detected in audio chunk")
                chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure audio is properly scaled for output (-1.0 to 1.0)
            max_val = np.max(np.abs(chunk))
            if max_val > 1.0:
                chunk = chunk / max_val
                
            # Apply a small fade-in to the first chunks to avoid clicks
            if self.frame_count < 10:
                fade_length = min(100, chunk.shape[0])
                fade_in = np.linspace(0, 1, fade_length)
                chunk[:fade_length] *= fade_in.reshape(-1, 1)
            
            output_data = chunk
            
        except Exception as e:
            logger.error(f"Error in callback: {e}")
            traceback.print_exc()
            # Return silence but continue playback
            output_data = np.zeros((frame_count, self.channels), dtype=np.float32)
        finally:
            if lock_acquired:
                self.buffer_lock.release()
        
        # For debugging audio output problems
        # CHANGE 19: Reduce debug logging for audio chunks
        if self.frame_count % LOG_AUDIO_CHUNK_EVERY == 0:  # Reduced from every 100 frames
            logger.debug(f"Audio chunk: {output_data.shape}, range: {np.min(output_data):.3f} to {np.max(output_data):.3f}")
            
        # Convert to bytes (required for PyAudio)
        try:
            output_bytes = output_data.flatten().tobytes()
            return (output_bytes, pyaudio.paContinue)
        except Exception as e:
            logger.error(f"Error converting output to bytes: {e}")
            return (np.zeros(frame_count * self.channels, dtype=np.float32).tobytes(), pyaudio.paContinue)

    def _update_fft(self, chunk):
        """Calculate FFT for visualization."""
        # Optimize FFT calculation for better visualization responsiveness
        if self.frame_count % FFT_PROCESS_EVERY != 0:  # Skip some frames but not too many
            return
            
        try:
            if chunk is None or len(chunk) == 0:
                logger.warning("Empty chunk passed to _update_fft")
                return
                
            # Use left channel for FFT
            data = chunk[:, 0] if self.channels > 1 and chunk.shape[1] > 0 else chunk
            
            # Apply a window function to reduce spectral leakage
            windowed_data = data * np.hanning(len(data))
            
            # Ensure we have enough data for FFT
            if len(windowed_data) < FFT_SIZE:
                padding = np.zeros(FFT_SIZE - len(windowed_data))
                windowed_data = np.concatenate((windowed_data, padding))
            elif len(windowed_data) > FFT_SIZE:
                # Take the most recent data for better responsiveness
                windowed_data = windowed_data[-FFT_SIZE:]
            
            # Calculate FFT - optimize for speed
            fft = np.abs(np.fft.rfft(windowed_data)) / len(windowed_data)
            
            # Scale and normalize (use square of FFT for energy)
            fft_squared = np.square(fft)
            fft_smoothed = fft_squared[:FFT_SIZE//2]
            
            # Apply logarithmic scaling for better visualization
            eps = 1e-10  # Small value to avoid log(0)
            log_fft = np.log10(fft_smoothed + eps)
            
            # Normalize to 0-1 range
            min_val = np.min(log_fft)
            max_val = np.max(log_fft)
            
            if max_val > min_val:
                normalized = (log_fft - min_val) / (max_val - min_val + eps)
            else:
                normalized = np.zeros_like(log_fft)
                
            # Add random noise if all values are zero to make visualization more interesting
            if not np.any(normalized > 0.01):
                normalized = normalized + np.random.random(normalized.shape) * 0.05
            
            # Apply some smoothing to avoid flickering - adjust smoothing factor for responsiveness
            if self.fft_data is not None:
                alpha = 0.6  # Reduced from 0.7 to make visualization more responsive
                normalized = alpha * self.fft_data + (1 - alpha) * normalized
            
            self.fft_data = normalized
            
            # Calculate RGB values from FFT data
            # Enhanced mapping for more vibrant colors
            low_band = np.mean(normalized[:len(normalized)//3])
            mid_band = np.mean(normalized[len(normalized)//3:2*len(normalized)//3])
            high_band = np.mean(normalized[2*len(normalized)//3:])
            
            # Apply non-linear scaling to make colors more vibrant
            r = int(255 * np.power(low_band, 0.7))  # More sensitive to low frequencies
            g = int(255 * np.power(mid_band, 0.8))
            b = int(255 * np.power(high_band, 0.75))  # More sensitive to high frequencies
            
            # Ensure there's always some color
            r = max(30, r)
            g = max(30, g)
            b = max(30, b)
            
            self.current_rgb = (
                min(255, r),
                min(255, g),
                min(255, b)
            )
            
            # Reduce debug information to only log every 200 frames
            if self.frame_count % 200 == 0:
                logger.debug(f"FFT updated - max value: {np.max(normalized):.4f}, RGB: {self.current_rgb}")
            
        except Exception as e:
            logger.error(f"Error in _update_fft: {e}")
            traceback.print_exc()
            # If there's an error, keep the previous data
            if self.fft_data is None:
                self.fft_data = np.zeros(FFT_SIZE // 2, dtype=np.float32)
                self.current_rgb = (50, 50, 50)  # Default gray

class AudioPlayerUI:
    """Simple GUI for audio playback and visualization."""
    def __init__(self):
        """Initialize the UI."""
        self.player = AudioPlayer()
        self.running = True
        self.clock = None
        self.font = None
        self.small_font = None
        self.screen = None
        self.width = 800
        self.height = 600
        self.ui_state = "file_browser"  # States: "file_browser", "playback"
        self.file_list = []
        self.current_dir = os.path.expanduser("~")
        self.scroll_offset = 0
        self.hover_index = -1
        self.file_filter = [".mp3", ".wav", ".ogg", ".flac"]
        self.last_click_time = 0
        self.last_status_update = 0
        self.status_message = ""
        self.status_color = (255, 255, 255)
        self.status_duration = 0
        
        # Visualization
        self.visualization_mode = "spectrum"  # "spectrum", "waveform", "rgb"
        self.last_frame_time = 0
        self.frame_times = []  # For FPS calculation
        
        # Audio directory shortcuts
        self.audio_dirs = [
            os.path.expanduser("~/Music"),
            os.path.expanduser("~/Documents"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../testWavs")
        ]

    def init_ui(self):
        """Initialize the UI."""
        pygame.init()
        pygame.display.set_caption("Simple Audio Player")
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        
        # Load fonts
        try:
            self.font = pygame.font.SysFont("Arial", 18)
            self.small_font = pygame.font.SysFont("Arial", 14)
            self.large_font = pygame.font.SysFont("Arial", 24)
        except Exception as e:
            logger.error(f"Error loading fonts: {e}")
            self.font = pygame.font.Font(None, 18)
            self.small_font = pygame.font.Font(None, 14)
            self.large_font = pygame.font.Font(None, 24)
        
        self.refresh_file_list()
        
        # Set initial status
        self.show_status("Ready to play - Select a file to begin", (220, 220, 220), 3)
        
        return True

    def refresh_file_list(self):
        """Refresh the file list for the current directory."""
        try:
            self.file_list = []
            
            # Add parent directory option
            if self.current_dir != os.path.expanduser("~"):
                self.file_list.append({
                    "name": "..",
                    "path": os.path.dirname(self.current_dir),
                    "type": "dir"
                })
            
            # Add shortcut directories
            for dir_path in self.audio_dirs:
                if os.path.isdir(dir_path) and os.path.normpath(dir_path) != os.path.normpath(self.current_dir):
                    dir_name = os.path.basename(dir_path)
                    if not dir_name:  # For root directories
                        dir_name = dir_path
                    self.file_list.append({
                        "name": f"→ {dir_name}",
                        "path": dir_path,
                        "type": "shortcut" 
                    })
            
            # Add directories and files
            entries = os.listdir(self.current_dir)
            for entry in sorted(entries):
                full_path = os.path.join(self.current_dir, entry)
                if os.path.isdir(full_path):
                    self.file_list.append({
                        "name": entry,
                        "path": full_path,
                        "type": "dir"
                    })
                elif os.path.isfile(full_path):
                    ext = os.path.splitext(entry)[1].lower()
                    if ext in self.file_filter:
                        self.file_list.append({
                            "name": entry,
                            "path": full_path,
                            "type": "file"
                        })
            
            self.scroll_offset = 0
            logger.info(f"Refreshed file list: {len(self.file_list)} entries in {self.current_dir}")
        except Exception as e:
            logger.error(f"Error refreshing file list: {e}")
            self.show_status(f"Error: {str(e)}", (255, 100, 100), 3)

    def run(self):
        """Run the UI."""
        if not self.init_ui():
            return False
        
        while self.running:
            current_time = time.time()
            
            # Handle events
            self.handle_events()
            
            # Clear screen
            self.screen.fill((30, 30, 30))
            
            # Render UI based on state
            if self.ui_state == "file_browser":
                self.render_file_browser()
            elif self.ui_state == "playback":
                self.render_playback()
            
            # Draw status message if needed
            if current_time - self.last_status_update < self.status_duration:
                self.draw_status()
            
            # Calculate FPS for debugging
            if current_time - self.last_frame_time > 0.5:
                if self.frame_times:
                    avg_fps = len(self.frame_times) / sum(self.frame_times)
                    pygame.display.set_caption(f"Simple Audio Player - FPS: {avg_fps:.1f}")
                self.frame_times = []
                self.last_frame_time = current_time
            
            # Update display
            pygame.display.flip()
            
            # Cap framerate to reduce CPU usage
            dt = self.clock.tick(MAX_FPS) / 1000.0
            self.frame_times.append(dt)
        
        # Clean up
        self.player.stop()
        pygame.quit()
        return True

    def handle_events(self):
        """Handle UI events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.ui_state == "playback":
                        self.ui_state = "file_browser"
                    else:
                        self.running = False
                elif event.key == pygame.K_SPACE:
                    if self.ui_state == "playback":
                        if self.player.is_playing():
                            self.player.stop()
                        else:
                            self.player.start()
            
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.size
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    self.scroll_offset = max(0, self.scroll_offset - 3)
                elif event.button == 5:  # Scroll down
                    self.scroll_offset = min(len(self.file_list) - 1, self.scroll_offset + 3)
                elif event.button == 1:  # Left click
                    self.handle_click(event.pos)
    
    def handle_click(self, pos):
        """Handle click events based on UI state."""
        if self.ui_state == "file_browser":
            # Check file list clicks
            item_height = 30
            visible_items = (self.height - 100) // item_height
            
            for i in range(min(visible_items, len(self.file_list) - self.scroll_offset)):
                index = i + self.scroll_offset
                item_rect = pygame.Rect(20, 80 + i * item_height, self.width - 40, item_height)
                
                if item_rect.collidepoint(pos):
                    self.handle_file_selection(index)
                    break
        
        elif self.ui_state == "playback":
            # Check play/stop button
            play_btn = pygame.Rect(self.width // 2 - 50, self.height - 60, 100, 40)
            if play_btn.collidepoint(pos):
                if self.player.is_playing():
                    logger.debug("Play/Stop button: Stopping playback")
                    self.player.stop()
                    self.show_status("Playback stopped", (220, 220, 100), 2)
                else:
                    logger.debug("Play/Stop button: Starting/resuming playback")
                    # If playback finished, reset position
                    if self.player.finished:
                        logger.debug("Playback was finished, resetting position to start")
                        self.player.position = 0
                        self.player.finished = False
                        
                    success = self.player.start()
                    if success:
                        self.show_status("Playback started", (100, 220, 100), 2)
                    else:
                        self.show_status("Error starting playback - see log", (255, 100, 100), 3)
            
            # Check back button
            back_btn = pygame.Rect(20, 20, 80, 30)
            if back_btn.collidepoint(pos):
                self.ui_state = "file_browser"
                self.player.stop()
    
    def handle_file_selection(self, index):
        """Handle file selection from the browser."""
        if 0 <= index < len(self.file_list):
            item = self.file_list[index]
            
            if item["type"] == "dir" or item["type"] == "shortcut":
                # Change directory
                if os.path.isdir(item["path"]):
                    self.current_dir = item["path"]
                    self.refresh_file_list()
                    self.show_status(f"Navigated to: {os.path.basename(item['path'])}", (100, 200, 255), 1)
                else:
                    self.show_status(f"Cannot open directory: {item['path']}", (255, 100, 100), 3)
            
            elif item["type"] == "file":
                # Load and play audio file
                file_path = item["path"]
                logger.info(f"Selected file: {file_path}")
                
                # Show loading status
                self.show_status(f"Loading {os.path.basename(file_path)}...", (100, 200, 255), 2)
                pygame.display.flip()  # Force update to show loading message
                
                if self.player.load_file(file_path):
                    self.ui_state = "playback"
                    self.player.start()
                    self.show_status(f"Playing {os.path.basename(file_path)}", (100, 255, 100), 2)
                else:
                    self.show_status("Error loading file", (255, 100, 100), 3)

    def render_file_browser(self):
        """Render the file browser UI."""
        # Draw title
        title_text = f"Audio Files: {os.path.basename(self.current_dir)}"
        title_surf = self.large_font.render(title_text, True, (220, 220, 220))
        self.screen.blit(title_surf, (20, 20))
        
        # Draw path
        path_text = self.current_dir
        path_surf = self.small_font.render(path_text, True, (180, 180, 180))
        self.screen.blit(path_surf, (20, 50))
        
        # Draw file list
        item_height = 30
        visible_items = (self.height - 100) // item_height
        
        for i in range(min(visible_items, len(self.file_list) - self.scroll_offset)):
            index = i + self.scroll_offset
            item = self.file_list[index]
            
            # Determine item color based on type
            if item["type"] == "dir":
                color = (100, 180, 255)
                prefix = "📁 "
            elif item["type"] == "shortcut":
                color = (100, 220, 180)
                prefix = "🔗 "
            else:
                color = (220, 220, 220)
                prefix = "🎵 "
            
            # Highlight on hover
            item_rect = pygame.Rect(20, 80 + i * item_height, self.width - 40, item_height)
            mouse_pos = pygame.mouse.get_pos()
            
            if item_rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.screen, (70, 70, 90), item_rect)
                self.hover_index = index
            elif self.hover_index == index:
                pygame.draw.rect(self.screen, (60, 60, 80), item_rect)
            
            # Draw item text
            text = f"{prefix}{item['name']}"
            item_surf = self.font.render(text, True, color)
            self.screen.blit(item_surf, (30, 85 + i * item_height))
        
        # Draw scrollbar if needed
        if len(self.file_list) > visible_items:
            scrollbar_height = (visible_items / len(self.file_list)) * (visible_items * item_height)
            scrollbar_pos = (self.scroll_offset / len(self.file_list)) * (visible_items * item_height)
            scrollbar_rect = pygame.Rect(self.width - 15, 80 + scrollbar_pos, 10, scrollbar_height)
            pygame.draw.rect(self.screen, (100, 100, 120), scrollbar_rect)
    
    def render_playback(self):
        """Render the playback UI."""
        # Draw back button
        back_btn = pygame.Rect(20, 20, 80, 30)
        pygame.draw.rect(self.screen, (80, 80, 100), back_btn)
        back_text = self.small_font.render("Back", True, (220, 220, 220))
        self.screen.blit(back_text, (35, 25))
        
        # Draw file name
        if self.player.file_name:
            name_text = self.font.render(self.player.file_name, True, (220, 220, 220))
            self.screen.blit(name_text, (self.width // 2 - name_text.get_width() // 2, 30))
        
        # Draw playback status
        if self.player.is_playing():
            status = "Playing"
            status_color = (100, 200, 100)
        elif self.player.finished:
            status = "Finished"
            status_color = (200, 200, 100)
            
            # Auto-restart with certain files if pattern needs it
            current_time = time.time()
            if current_time - self.last_status_update > 5.0:  # Only check every 5 seconds
                self.last_status_update = current_time
                if self.player.file_name and self.player.file_name.endswith(('.wav', '.mp3')):
                    if not self.player.is_playing() and self.player.finished:
                        logger.debug("Auto-restart check: Playback finished, could restart")
                        # Option to auto-restart could be added here
        else:
            status = "Stopped"
            status_color = (200, 100, 100)
            
        status_text = self.font.render(status, True, status_color)
        self.screen.blit(status_text, (self.width // 2 - status_text.get_width() // 2, 70))
        
        # Draw progress bar - safely handle progress calculation
        try:
            progress = self.player.get_progress()
        except Exception as e:
            logger.error(f"Error getting progress: {e}")
            progress = 0
            
        progress_rect = pygame.Rect(20, 120, self.width - 40, 20)
        pygame.draw.rect(self.screen, (60, 60, 70), progress_rect)
        if progress > 0:
            filled_rect = pygame.Rect(20, 120, int((self.width - 40) * progress), 20)
            pygame.draw.rect(self.screen, (100, 180, 255), filled_rect)
        
        # Draw play/stop button with correct label
        if self.player.is_playing():
            btn_color = (220, 100, 100)  # Stop button (red)
            btn_text = "Stop"
        elif self.player.finished:
            btn_color = (100, 180, 100)  # Play button (green) for restart
            btn_text = "Restart" 
        else:
            btn_color = (100, 220, 100)  # Play button (green)
            btn_text = "Play"
            
        play_btn = pygame.Rect(self.width // 2 - 50, self.height - 60, 100, 40)
        pygame.draw.rect(self.screen, btn_color, play_btn)
        btn_surf = self.font.render(btn_text, True, (30, 30, 30))
        self.screen.blit(btn_surf, (self.width // 2 - btn_surf.get_width() // 2, self.height - 50))
        
        # Draw visualization
        self.render_visualization()
        
        # Handle finished playback state
        now = time.time()
        if not self.player.is_playing() and self.player.finished and now - self.last_status_update > 2.0:
            # Only update status if it's been a while since last update
            if now - self.last_status_update > 10:  # Only update every 10 seconds at most
                progress = self.player.get_progress()
                if progress > 0.9:  
                    # Playback completed normally
                    self.show_status("Playback finished", (220, 220, 100), 2)
                elif progress < 0.1:
                    # Very little progress was made - likely an error
                    self.show_status("Audio playback error - check logs", (255, 100, 100), 3)
    
    def render_visualization(self):
        """Render audio visualization based on actual audio data."""
        viz_rect = pygame.Rect(20, 150, self.width - 40, self.height - 220)
        
        # Initialize FFT data if None
        if self.player.fft_data is None:
            self.player.fft_data = np.zeros(FFT_SIZE // 2, dtype=np.float32) 
            self.player.current_rgb = (50, 50, 50)
        
        # Check if player has valid data to visualize
        has_data = np.any(self.player.fft_data > 0.01)
            
        # Spectrum visualization (FFT)
        if has_data:
            # Get the current RGB color for background tint
            r, g, b = self.player.current_rgb
            bg_color = (max(30, r // 8), max(30, g // 8), max(30, b // 8))
            
            # Draw background with slight tint
            pygame.draw.rect(self.screen, bg_color, viz_rect)
            
            # Make a copy of the FFT data to prevent threading issues
            fft_data = np.copy(self.player.fft_data)
            
            # Draw FFT bars
            bar_width = max(2, (viz_rect.width // len(fft_data) - 1))
            bar_count = min(len(fft_data), viz_rect.width // (bar_width + 1))
            
            for i in range(bar_count):
                # Skip some bands for lower frequencies to make visualization more interesting
                if bar_count > 150:
                    idx = int(i ** 1.3) % len(fft_data)
                else:
                    idx = i % len(fft_data)
                
                # Get height from FFT data - ensure it's positive and scale for visibility
                height = int(max(1, fft_data[idx] * viz_rect.height))
                
                # Calculate bar color (gradient based on frequency)
                hue = 240 - (240 * i / bar_count)  # Blue to red
                bar_color = self.hsv_to_rgb(hue / 360, 0.8, 0.9)
                
                # Apply volume intensity
                intensity = max(0.3, min(1.0, height / (viz_rect.height * 0.7)))
                r, g, b = bar_color
                bar_color = (
                    int(r * intensity),
                    int(g * intensity),
                    int(b * intensity)
                )
                
                # Draw the bar
                bar_rect = pygame.Rect(
                    viz_rect.left + i * (bar_width + 1),
                    viz_rect.bottom - height,
                    bar_width,
                    height
                )
                pygame.draw.rect(self.screen, bar_color, bar_rect)
            
            # Draw a colored circle representing the current sound
            r, g, b = self.player.current_rgb
            glow_size = 40 + int(sum(self.player.current_rgb) / 10)
            center_x = viz_rect.right - glow_size
            center_y = viz_rect.top + glow_size
            
            # Draw glow effect
            for i in range(3):
                size = glow_size - i * 10
                alpha = 100 - i * 30
                glow_color = (r, g, b, alpha)
                glow_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, glow_color, (size, size), size)
                self.screen.blit(glow_surf, (center_x - size, center_y - size))
        else:
            # Just draw an empty visualization area
            pygame.draw.rect(self.screen, (40, 40, 50), viz_rect)
            
            if not self.player.is_playing():
                text = "Play audio to see visualization"
                text_surf = self.font.render(text, True, (140, 140, 160))
                text_rect = text_surf.get_rect(center=(viz_rect.centerx, viz_rect.centery))
                self.screen.blit(text_surf, text_rect)
            elif self.player.is_playing():
                text = "Audio data processing..."
                text_surf = self.font.render(text, True, (180, 180, 200))
                text_rect = text_surf.get_rect(center=(viz_rect.centerx, viz_rect.centery))
                self.screen.blit(text_surf, text_rect)
    
    def hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB color."""
        if s == 0.0:
            return (int(v * 255), int(v * 255), int(v * 255))
        
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i %= 6
        
        if i == 0:
            return (int(v * 255), int(t * 255), int(p * 255))
        elif i == 1:
            return (int(q * 255), int(v * 255), int(p * 255))
        elif i == 2:
            return (int(p * 255), int(v * 255), int(t * 255))
        elif i == 3:
            return (int(p * 255), int(q * 255), int(v * 255))
        elif i == 4:
            return (int(t * 255), int(p * 255), int(v * 255))
        else:
            return (int(v * 255), int(p * 255), int(q * 255))
    
    def show_status(self, message, color=(255, 255, 255), duration=2):
        """Show a status message."""
        self.status_message = message
        self.status_color = color
        self.status_duration = duration
        self.last_status_update = time.time()
        logger.info(f"Status: {message}")
    
    def draw_status(self):
        """Draw the status message."""
        if not self.status_message:
            return
            
        # Create a background with alpha
        status_height = 30
        status_surf = pygame.Surface((self.width, status_height), pygame.SRCALPHA)
        status_surf.fill((30, 30, 30, 200))
        
        # Render text
        text_surf = self.font.render(self.status_message, True, self.status_color)
        text_rect = text_surf.get_rect(center=(self.width // 2, status_height // 2))
        status_surf.blit(text_surf, text_rect)
        
        # Draw at bottom of screen
        self.screen.blit(status_surf, (0, self.height - status_height))

def main():
    """Run the application."""
    parser = argparse.ArgumentParser(description='Audio Visualizer')
    parser.add_argument('--input-file', '-i', help='Path to input audio file')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    parser.add_argument('--autoplay', '-a', action='store_true', help='Automatically play the input file')
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Check PyAudio availability
    if not HAS_PYAUDIO:
        logger.error("PyAudio is required but not available. Please install with: pip install pyaudio")
        print("ERROR: PyAudio is required but not available.")
        print("Install with: pip install pyaudio")
        print("For Linux, you may need to install portaudio19-dev first:")
        print("sudo apt-get install portaudio19-dev")
        return 1

    # Create UI
    ui = AudioPlayerUI()

    # Check input file
    if args.input_file:
        input_file = Path(args.input_file)
        if not input_file.exists():
            logger.error(f"Input file not found: {args.input_file}")
            return 1
        
        logger.info(f"Using input file: {args.input_file}")
        if ui.player.load_file(args.input_file):
            ui.ui_state = "playback"  # Switch to playback mode immediately
            
            # Auto-play if flag is set
            if args.autoplay:
                logger.info("Autoplay enabled - starting playback automatically")
                ui.player.start()
        else:
            logger.error(f"Failed to load input file: {args.input_file}")
            return 1

    # Start UI
    ui.run()
    return 0

if __name__ == "__main__":
    main() 