#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the Audio-LED project
# Audio-LED provides audio reactive lighting effects for Raspberry Pi/Arduino/ESP projects
# https://github.com/aluhmann/audio-led/
# 
# Original elements from:
# https://github.com/scottlawsonbc/audio-reactive-led-strip
# Copyright(c) 2017 Scott Lawson
# 
# This file may be licensed under the terms of the
# GNU General Public License Version 3 (the ``GPL'').
#
# Audio-LED is free software: you can redistribute it and/or modify
# it under the terms of the GPL.
#
# Audio-LED is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GPL for more details.
# 
# You should have received a copy of the GPL License
# along with Audio-LED.  If not, see <http://www.gnu.org/licenses/>.

# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import time
import logging
import threading
from collections import deque
from io import BytesIO
import traceback

import numpy as np
import pyaudio
from pydub import AudioSegment
import scipy.signal as signal

# Conditionally import soundfile - make it optional
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("soundfile module not found, falling back to pydub for audio file loading")

# Conditionally import librosa - make it optional
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("librosa module not found, using simpler resampling methods")

# Conditionally import MP3Player - create a fallback if not available
try:
    from audio_led.audio.mp3_player import MP3Player
    MP3PLAYER_AVAILABLE = True
except ImportError:
    MP3PLAYER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("mp3_player module not found, using pydub for MP3 playback")
    
    # Create a simple fallback MP3Player using pydub directly
    class MP3Player:
        """Fallback MP3 player implementation"""
        def __init__(self, file_path):
            self.file_path = file_path
            self.position = 0
            self.data = None
            self.sample_rate = 44100
            self.segment = None
            self._load()
            
        def _load(self):
            try:
                self.segment = AudioSegment.from_file(self.file_path, format="mp3")
                self.sample_rate = self.segment.frame_rate
                # Convert to mono if stereo
                if self.segment.channels > 1:
                    self.segment = self.segment.set_channels(1)
                # Convert to numpy array
                self.data = np.array(self.segment.get_array_of_samples())
                return True
            except Exception as e:
                logger.error("Error loading MP3: %s", str(e))
                return False
                
        def read_chunk(self, chunk_size):
            if self.data is None:
                return None
            if self.position + chunk_size <= len(self.data):
                chunk = self.data[self.position:self.position + chunk_size]
                self.position += chunk_size
                return chunk
            else:
                return None
                
        def rewind(self):
            self.position = 0

from audio_led.common.utils import resample_frame, get_elapsed, map_range, clamp
from audio_led.common.thread_manager import ThreadManager

# Initialize the logger
logger = logging.getLogger(__name__)

# Constants
INPUT_NONE = 0
INPUT_FILE = 1
INPUT_STREAM = 2
INPUT_MIC = 3

BUFFER_FACTOR = 4
BUFFER_MIN_READ = 20
MAX_BUFFER_SIZE = 16384
MAX_HISTORY_SIZE = 2048
SILENCE_SAMPLES = 1024
SILENCE_CHECK_MIN = 650

MAX_RESAMPLE_RATIO = 0.45
MIN_RESAMPLE_RATIO = 0.05

# Dictionary mapping file extensions to audio formats
# Using lowercase keys for case-insensitive comparison
AUDIO_FORMAT_MAP = {
    'wav': 'wav',
    'mp3': 'mp3',
    'flac': 'flac',
    'ogg': 'ogg'
}

# Audio cache for resampled data
RESAMPLED_AUDIO_CACHE = {}
# Maximum number of entries in the cache
MAX_CACHE_ENTRIES = 10
# Cached audio data size in bytes
CACHE_SIZE = 0
# Maximum cache size in bytes (default: 500MB)
MAX_CACHE_SIZE = 500 * 1024 * 1024

def get_file_format(filename):
    """
    Detect audio file format from extension or file header
    
    Parameters
    ----------
    filename : str
        Path to the audio file
    
    Returns
    -------
    str
        Detected format ('mp3', 'wav', 'flac', 'ogg', or None)
    """
    # First check by extension
    if filename.startswith('http'):
        # For URLs, we can only check by extension
        ext = os.path.splitext(filename.split('?')[0])[1].lower().lstrip('.')
        return AUDIO_FORMAT_MAP.get(ext)
    
    # For local files, check extension first
    ext = os.path.splitext(filename)[1].lower().lstrip('.')
    if ext in AUDIO_FORMAT_MAP:
        return AUDIO_FORMAT_MAP[ext]
    
    # If extension doesn't match or is missing, check file header
    try:
        with open(filename, 'rb') as f:
            header = f.read(12)
            # Check for WAV header (RIFF)
            if header.startswith(b'RIFF') and b'WAVE' in header:
                return 'wav'
            # Check for MP3 header (ID3 or MPEG frame sync)
            elif header.startswith(b'ID3') or (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
                return 'mp3'
            # Check for FLAC header
            elif header.startswith(b'fLaC'):
                return 'flac'
            # Check for OGG header
            elif header.startswith(b'OggS'):
                return 'ogg'
    except Exception as e:
        logger.error("Error detecting file format by header: %s", str(e))
    
    return None

def clear_cache_if_needed(new_data_size=0):
    """
    Clear audio cache if it exceeds size limits
    
    Parameters
    ----------
    new_data_size : int
        Size of new data to be added to cache
    """
    global CACHE_SIZE, RESAMPLED_AUDIO_CACHE
    
    # If adding this data would exceed max cache size
    if CACHE_SIZE + new_data_size > MAX_CACHE_SIZE:
        logger.info("Cache size limit reached (%d MB), clearing oldest entries", 
                    MAX_CACHE_SIZE // (1024 * 1024))
        
        # Sort cache entries by last access time
        sorted_entries = sorted(
            [(k, v.get('last_access', 0)) for k, v in RESAMPLED_AUDIO_CACHE.items()],
            key=lambda x: x[1]
        )
        
        # Remove oldest entries until we have enough space
        freed_space = 0
        for key, _ in sorted_entries:
            if freed_space >= new_data_size or len(RESAMPLED_AUDIO_CACHE) <= MAX_CACHE_ENTRIES // 2:
                break
                
            if key in RESAMPLED_AUDIO_CACHE:
                entry_size = RESAMPLED_AUDIO_CACHE[key].get('size', 0)
                del RESAMPLED_AUDIO_CACHE[key]
                freed_space += entry_size
                CACHE_SIZE -= entry_size
                logger.debug("Removed cache entry: %s, freed %d MB", 
                           key, entry_size // (1024 * 1024))
        
        # If we still have too many entries, remove more
        if len(RESAMPLED_AUDIO_CACHE) > MAX_CACHE_ENTRIES:
            # Remove oldest entries
            for key, _ in sorted_entries[:len(RESAMPLED_AUDIO_CACHE) - MAX_CACHE_ENTRIES]:
                if key in RESAMPLED_AUDIO_CACHE:
                    entry_size = RESAMPLED_AUDIO_CACHE[key].get('size', 0)
                    del RESAMPLED_AUDIO_CACHE[key]
                    CACHE_SIZE -= entry_size

class AudioInputHandler(object):
    """Class for handling audio inputs from various sources"""

    def __init__(self, config, device_id=None, skip_setup=False):
        """Initialize audio processor

        Parameters
        ----------
        config : dict
            Configuration dictionary with audio settings
        device_id : int
            Device ID for PyAudio
        skip_setup : bool
            If True, skip initial setup
        """
        logger.debug("Initializing audio input handler...")
        # Initialize variables
        self.frames = None
        self.current_chunk = None
        self.input_type = None
        self.fft_range_norm = None
        self.audio_range_norm = None
        self.stream = None
        self.chunk_count = 0
        self.wav_data = None
        self.wav_idx = 0
        self.wav_stream = None
        self.input_thread = None
        self.chunk_ready = False
        self.chunk_available = None
        self.buffer = None
        self._data_buffer = None
        self._mp3_player = None
        self._input_file = None
        self._input_file_name = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread_running = False
        self._is_mp3 = False
        self._is_wav = False
        self._is_ogg = False
        self._is_flac = False
        self._py_audio = None
        self._is_mic = False
        self._pause_time = 0
        self._start_time = 0
        self._end_time = 0
        self._running = False
        self._audio_finished = False
        self.sample_rate = config.get("SAMPLE_RATE", 48000)
        self.device_id = device_id if device_id is not None else config.get("DEVICE_ID", None)
        self.device_name = None
        self.device_sample_rate = 0
        self.format_bytes = 0
        self.chunk_size = config.get("CHUNK_SIZE", 1024)
        self.bytes_per_sample = config.get("BYTES_PER_SAMPLE", 2)
        self.mic_rate = config.get("MIC_RATE", 48000)
        self.output_rate = self.mic_rate
        self.buffer_size = config.get("BUFFER_SIZE", self.chunk_size * BUFFER_FACTOR)
        self.channels = config.get("CHANNELS", 1)
        self.frames_per_buffer = config.get("FRAMES_PER_BUFFER", self.chunk_size)
        self.resample_buff_ratio = 1.0
        self.chunks_processed = 0
        self.loop = config.get("LOOP", True)
        self.min_frequency = config.get("MIN_FREQUENCY", 20)
        self.max_frequency = config.get("MAX_FREQUENCY", 16000)
        self.min_volume_threshold = config.get("MIN_VOLUME_THRESHOLD", 1e-7)
        self.start_volume_threshold = config.get("START_VOLUME_THRESHOLD", 1e-5)
        self.history_size = config.get("HISTORY_SIZE", 1)
        self.history = deque(maxlen=self.history_size)
        self.config = config
        self.use_fft_ranges = False
        self.initialized = False

        # History deques
        self.history_mel = []
        self.history_mel_raw = []
        self.history_vol = []
        self.history_vol_raw = []
        self.history_mel_smoothed = []
        self.history_vol_smoothed = []
        
        if not skip_setup:
            self.setup()

    def setup(self):
        """Set up the audio input handler"""
        logger.debug("Setting up audio input handler...")
        self.setup_input_type(self.device_id)
        
    def setup_input_type(self, device_id=None):
        """Set up the audio input type based on device_id

        Parameters
        ----------
        device_id : int
            Device ID for PyAudio
        """
        if device_id is None:
            logger.debug("No device ID specified")
            self.input_type = INPUT_NONE
            return
        
        self.device_id = device_id
        logger.info("Setting up input type for device ID: %s", device_id)
        
        # Try to convert to integer for numeric IDs
        if isinstance(device_id, str) and device_id.isdigit():
            device_id = int(device_id)
        
        # If device is a string, try to handle it as a file path
        if isinstance(device_id, str) and (os.path.isfile(device_id) or device_id.startswith('http')):
            # Set file input
            self._input_file = device_id
            self._input_file_name = os.path.basename(device_id) if not device_id.startswith('http') else device_id
            self.input_type = INPUT_FILE
            logger.info("Input set to file: %s", self._input_file_name)
            
            # Detect file format using the helper function
            file_format = get_file_format(self._input_file)
            
            # Set format flags based on detection result
            self._is_mp3 = file_format == 'mp3'
            self._is_wav = file_format == 'wav'
            self._is_flac = file_format == 'flac'
            self._is_ogg = file_format == 'ogg'
            
            if file_format:
                logger.info("Detected audio format: %s", file_format)
            else:
                logger.warning("Unknown audio format for file: %s", self._input_file_name)
                # We'll try to load it anyway with pydub as a fallback
        
        # Handle microphone or line input
        elif isinstance(device_id, (int, str)) and not (isinstance(device_id, str) and device_id.isdigit()):
            # Set mic or stream input
            if device_id == 'mic' or isinstance(device_id, int):
                self._is_mic = True
                mic_id = 0 if device_id == 'mic' else device_id
                self.input_type = INPUT_MIC
                self.device_id = mic_id
                logger.info("Input set to microphone with ID: %s", self.device_id)
            else:
                self.input_type = INPUT_STREAM
                self.device_id = device_id
                logger.info("Input set to audio stream: %s", self.device_id)
        else:
            # Invalid device ID
            logger.warning("Invalid device ID: %s", device_id)
            self.input_type = INPUT_NONE
            return
            
    def setup_pyaudio_stream(self):
        """Set up the PyAudio stream for mic input"""
        if self._py_audio is not None:
            self.close_pyaudio_stream()
            
        # Set up pyaudio
        self._py_audio = pyaudio.PyAudio()
        
        # Get device info
        device_info = None
        sample_rate = self.mic_rate
        
        # Find device info
        try:
            if isinstance(self.device_id, int):
                device_info = self._py_audio.get_device_info_by_index(self.device_id)
                self.device_name = device_info.get('name')
                logger.info("Using audio device: %s", self.device_name)
                # Try to get the supported sample rate
                try:
                    # Some devices report sample rates as strings, some as ints
                    sr = device_info.get('defaultSampleRate')
                    if sr:
                        if isinstance(sr, str) and sr.isdigit():
                            sample_rate = int(sr)
                        elif isinstance(sr, (int, float)):
                            sample_rate = int(sr)
                    self.device_sample_rate = sample_rate
                    logger.info("Device sample rate detected: %d Hz", self.device_sample_rate)
                except (ValueError, TypeError) as e:
                    logger.warning("Error parsing sample rate: %s", str(e))
                    # Use default
                    self.device_sample_rate = self.mic_rate
                    logger.info("Using default sample rate: %d Hz", self.device_sample_rate)
            else:
                # Try to find device by string name
                for i in range(self._py_audio.get_device_count()):
                    info = self._py_audio.get_device_info_by_index(i)
                    if self.device_id.lower() in info['name'].lower():
                        device_info = info
                        self.device_id = i
                        self.device_name = info.get('name')
                        # Get sample rate
                        try:
                            sr = device_info.get('defaultSampleRate')
                            if sr:
                                if isinstance(sr, str) and sr.isdigit():
                                    sample_rate = int(sr)
                                elif isinstance(sr, (int, float)):
                                    sample_rate = int(sr)
                            self.device_sample_rate = sample_rate
                            logger.info("Device sample rate detected: %d Hz", self.device_sample_rate)
                        except (ValueError, TypeError) as e:
                            logger.warning("Error parsing sample rate: %s", str(e))
                            # Use default
                            self.device_sample_rate = self.mic_rate
                            logger.info("Using default sample rate: %d Hz", self.device_sample_rate)
                        break
        except Exception as e:
            logger.error("Error finding audio device: %s", str(e))
            self.device_sample_rate = self.mic_rate
            logger.info("Using default sample rate: %d Hz", self.device_sample_rate)
            
        # If device info not found, use default
        if not device_info:
            logger.warning("Device info not found, using default device")
            self.device_sample_rate = self.mic_rate
            logger.info("Using default sample rate: %d Hz", self.device_sample_rate)
            
        # Setup audio stream
        try:
            self.stream = self._py_audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=sample_rate,
                input=True,
                input_device_index=self.device_id if isinstance(self.device_id, int) else None,
                frames_per_buffer=self.frames_per_buffer
            )
            logger.info("Opened pyaudio stream with sample rate: %d Hz", sample_rate)
        except Exception as e:
            logger.error("Error opening audio stream: %s", str(e))
            # Try to fall back to default device if specific device fails
            try:
                logger.info("Trying to open default audio device")
                self.stream = self._py_audio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.mic_rate,  # Use default rate for fallback
                    input=True,
                    frames_per_buffer=self.frames_per_buffer
                )
                sample_rate = self.mic_rate
                logger.info("Opened default audio device with sample rate: %d Hz", sample_rate)
            except Exception as e2:
                logger.error("Error opening default audio device: %s", str(e2))
                self.stream = None
                return False
        
        # Set output rate to device sample rate
        self.output_rate = sample_rate
        return True
        
    def close_pyaudio_stream(self):
        """Close the PyAudio stream"""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error("Error closing audio stream: %s", str(e))
            finally:
                self.stream = None
                
        if self._py_audio:
            try:
                self._py_audio.terminate()
            except Exception as e:
                logger.error("Error terminating PyAudio: %s", str(e))
            finally:
                self._py_audio = None
                
    def load_audio_file(self):
        """Load audio data from file"""
        global CACHE_SIZE
        
        logger.info("Loading audio file: %s", self._input_file_name)
        
        cache_key = f"{self._input_file}_{self.sample_rate}"
        
        # Check if we already have this file loaded and cached
        if cache_key in RESAMPLED_AUDIO_CACHE:
            logger.info("Using cached audio data for %s at %d Hz", 
                       self._input_file_name, self.sample_rate)
            
            # Update last access time
            RESAMPLED_AUDIO_CACHE[cache_key]['last_access'] = time.time()
            
            self.wav_data = RESAMPLED_AUDIO_CACHE[cache_key]['data']
            self.output_rate = RESAMPLED_AUDIO_CACHE[cache_key]['rate']
            return True
            
        # MP3 handling
        if self._is_mp3:
            logger.debug("Loading MP3 file using MP3Player")
            try:
                self._mp3_player = MP3Player(self._input_file)
                # Update the sample rate to the MP3's actual rate
                file_sample_rate = self._mp3_player.sample_rate
                logger.info("MP3 sample rate: %d Hz", file_sample_rate)
                self.output_rate = file_sample_rate
                return True
            except Exception as e:
                logger.error("Error loading MP3 file: %s", str(e))
                self._mp3_player = None
                # Try fallback to pydub
                return self._load_with_pydub()
        
        # WAV, FLAC, OGG handling with soundfile (if available)
        elif (self._is_wav or self._is_flac or self._is_ogg) and SOUNDFILE_AVAILABLE:
            try:
                # Using soundfile for better format support
                format_name = "WAV" if self._is_wav else "FLAC" if self._is_flac else "OGG"
                logger.info("Loading %s file using soundfile", format_name)
                
                # Load audio data
                audio_data, file_sample_rate = sf.read(self._input_file, dtype='float32')
                logger.info("%s sample rate: %d Hz", format_name, file_sample_rate)
                
                # Convert stereo to mono if needed
                if audio_data.ndim > 1 and audio_data.shape[1] > 1:
                    logger.info("Converting stereo to mono")
                    audio_data = np.mean(audio_data, axis=1)
                
                # Normalize the audio data to the range [-1, 1]
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                # Resample if needed
                if file_sample_rate != self.sample_rate:
                    logger.info("Resampling %s from %d Hz to %d Hz", 
                               format_name, file_sample_rate, self.sample_rate)
                    
                    # Using librosa for high-quality resampling if available
                    if LIBROSA_AVAILABLE:
                        audio_data = librosa.resample(
                            audio_data, 
                            orig_sr=file_sample_rate, 
                            target_sr=self.sample_rate
                        )
                    else:
                        # Simple resampling using scipy if librosa not available
                        audio_data = self._resample_with_scipy(audio_data, file_sample_rate, self.sample_rate)
                
                # Convert to 16-bit integers
                audio_data = (audio_data * 32767).astype(np.int16)
                
                # Calculate size for cache management
                data_size = audio_data.nbytes
                
                # Clear cache if needed
                if 'clear_cache_if_needed' in globals():
                    clear_cache_if_needed(data_size)
                
                # Store the resampled data
                self.wav_data = audio_data
                self.output_rate = self.sample_rate
                
                # Cache the resampled audio
                RESAMPLED_AUDIO_CACHE[cache_key] = {
                    'data': audio_data,
                    'rate': self.sample_rate,
                    'size': data_size,
                    'last_access': time.time()
                }
                
                # Update total cache size
                CACHE_SIZE += data_size
                
                return True
            except Exception as e:
                logger.error("Error loading %s file with soundfile: %s", format_name, str(e))
                # Try fallback to pydub
                return self._load_with_pydub()
        
        # Try using pydub for all other cases
        else:
            return self._load_with_pydub()
    
    def _load_with_pydub(self):
        """Load audio file using pydub as a fallback method"""
        try:
            logger.info("Loading %s with pydub", self._input_file_name)
            
            # Try to load the file with pydub
            try:
                audio = AudioSegment.from_file(self._input_file)
            except Exception as e:
                logger.error("Error loading with pydub: %s", str(e))
                
                # If loading fails, check for missing ffmpeg/avconv
                if "Couldn't find ffmpeg" in str(e) or "Couldn't find avconv" in str(e):
                    logger.error("Missing ffmpeg/avconv. Please install: 'sudo apt install ffmpeg' or 'sudo pacman -S ffmpeg'")
                
                return False
                
            file_sample_rate = audio.frame_rate
            logger.info("Audio file sample rate: %d Hz", file_sample_rate)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            # Convert stereo to mono if needed
            if audio.channels > 1:
                logger.info("Converting stereo to mono")
                samples = samples.reshape((-1, audio.channels))
                samples = np.mean(samples, axis=1)
            
            # Resample if needed
            if file_sample_rate != self.sample_rate:
                logger.info("Resampling audio from %d Hz to %d Hz", 
                           file_sample_rate, self.sample_rate)
                
                # Normalize to float
                float_samples = samples.astype(np.float32)
                if samples.dtype in [np.int16, np.int32]:
                    float_samples = float_samples / 32767.0
                
                # Resample using best available method
                if LIBROSA_AVAILABLE:
                    resampled = librosa.resample(
                        float_samples, 
                        orig_sr=file_sample_rate, 
                        target_sr=self.sample_rate
                    )
                else:
                    # Use scipy if librosa is not available
                    resampled = self._resample_with_scipy(float_samples, file_sample_rate, self.sample_rate)
                
                # Convert back to int16
                samples = (resampled * 32767).astype(np.int16)
            
            # Calculate size for cache management
            data_size = samples.nbytes
            
            # Clear cache if needed
            if 'clear_cache_if_needed' in globals():
                clear_cache_if_needed(data_size)
            
            # Store the resampled data
            self.wav_data = samples
            self.output_rate = self.sample_rate
            
            # Cache the resampled audio
            cache_key = f"{self._input_file}_{self.sample_rate}"
            RESAMPLED_AUDIO_CACHE[cache_key] = {
                'data': samples,
                'rate': self.sample_rate,
                'size': data_size,
                'last_access': time.time()
            }
            
            # Update total cache size
            global CACHE_SIZE
            CACHE_SIZE += data_size
            
            return True
        except Exception as e:
            logger.error("Error in pydub fallback: %s", str(e))
            return False
    
    def _resample_with_scipy(self, audio_data, orig_sr, target_sr):
        """Resample audio using scipy as a fallback when librosa is not available
        
        Parameters
        ----------
        audio_data : numpy.ndarray
            Audio data to resample
        orig_sr : int
            Original sample rate
        target_sr : int
            Target sample rate
            
        Returns
        -------
        numpy.ndarray
            Resampled audio data
        """
        logger.info("Using scipy for resampling")
        
        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        
        # Calculate output length
        output_length = int(len(audio_data) * ratio)
        
        # Use scipy's resample function
        resampled = signal.resample(audio_data, output_length)
        
        return resampled
        
    def read_audio_stream(self):
        """Read a chunk from the audio stream"""
        if self.input_type == INPUT_MIC and self.stream:
            try:
                data = self.stream.read(self.chunk_size)
                return np.frombuffer(data, dtype=np.int16)
            except Exception as e:
                logger.error("Error reading from audio stream: %s", str(e))
                return np.zeros(self.chunk_size, dtype=np.int16)
        return None
        
    def read_audio_file_chunk(self):
        """Read a chunk from the audio file"""
        if self.input_type != INPUT_FILE:
            return None
            
        # MP3 handling
        if self._is_mp3 and self._mp3_player:
            try:
                data = self._mp3_player.read_chunk(self.chunk_size)
                if data is None or len(data) == 0:
                    if self.loop and not self._audio_finished:
                        logger.info("End of MP3 file reached, looping...")
                        self._mp3_player.rewind()
                        data = self._mp3_player.read_chunk(self.chunk_size)
                    else:
                        logger.info("End of MP3 file reached")
                        self._audio_finished = True
                        return np.zeros(self.chunk_size, dtype=np.int16)
                return data
            except Exception as e:
                logger.error("Error reading MP3 chunk: %s", str(e))
                return np.zeros(self.chunk_size, dtype=np.int16)
                
        # WAV/FLAC/OGG handling
        elif self.wav_data is not None:
            if self.wav_idx + self.chunk_size <= len(self.wav_data):
                # Read the next chunk
                chunk = self.wav_data[self.wav_idx:self.wav_idx + self.chunk_size]
                self.wav_idx += self.chunk_size
                return chunk
            else:
                # End of file
                if self.loop and not self._audio_finished:
                    logger.info("End of audio file reached, looping...")
                    self.wav_idx = 0
                    return self.read_audio_file_chunk()
                else:
                    logger.info("End of audio file reached")
                    self._audio_finished = True
                    return np.zeros(self.chunk_size, dtype=np.int16)
        
        return None
        
    def input_thread_function(self):
        """Thread function for processing audio input"""
        logger.info("Starting audio input thread")
        self._thread_running = True
        
        # Initialize buffer
        if self._data_buffer is None:
            self._data_buffer = deque(maxlen=MAX_BUFFER_SIZE)
            
        self._start_time = time.time()
        self._running = True
        
        # Main processing loop
        while not self._stop_event.is_set() and self._running:
            # Check if paused
            if self._pause_event.is_set():
                time.sleep(0.01)
                continue
                
            # Process based on input type
            if self.input_type == INPUT_MIC:
                # Process microphone input
                chunk = self.read_audio_stream()
                if chunk is not None and len(chunk) > 0:
                    with self._lock:
                        self._data_buffer.append(chunk)
                        self.chunks_processed += 1
            elif self.input_type == INPUT_FILE:
                # Process file input
                chunk = self.read_audio_file_chunk()
                if chunk is not None:
                    with self._lock:
                        self._data_buffer.append(chunk)
                        self.chunks_processed += 1
                        
                    # Sleep to simulate real-time playback
                    # Calculate time to sleep based on chunk size and sample rate
                    sleep_time = self.chunk_size / self.output_rate
                    time.sleep(sleep_time * 0.9)  # Slightly less to account for processing
            
            # Check if we have enough data in the buffer
            if len(self._data_buffer) > BUFFER_MIN_READ:
                # Signal that a chunk is ready
                self.chunk_ready = True
            
            # Check if audio file has finished and we've processed all chunks
            if self._audio_finished and len(self._data_buffer) == 0:
                logger.info("Audio file finished and buffer empty")
                self._running = False
                
        # End of thread
        self._end_time = time.time()
        logger.info("Audio input thread finished. Runtime: %.2f seconds", 
                   self._end_time - self._start_time)
        self._thread_running = False
        
    def start(self):
        """Start the audio input handler"""
        if self._thread_running:
            logger.warning("Audio input thread already running")
            return False
            
        # Reset flags
        self._stop_event.clear()
        self._pause_event.clear()
        self._audio_finished = False
        self.chunk_ready = False
        self.chunks_processed = 0
        self.wav_idx = 0
        
        # Setup based on input type
        if self.input_type == INPUT_MIC:
            # Setup PyAudio for microphone input
            if not self.setup_pyaudio_stream():
                logger.error("Failed to setup PyAudio stream")
                return False
        elif self.input_type == INPUT_FILE:
            # Load audio file
            if not self.load_audio_file():
                logger.error("Failed to load audio file")
                return False
        else:
            logger.error("Invalid input type: %s", self.input_type)
            return False
            
        # Start the input thread
        self.input_thread = threading.Thread(target=self.input_thread_function)
        self.input_thread.daemon = True
        self.input_thread.start()
        
        logger.info("Audio input handler started with input type: %s", self.input_type)
        self.initialized = True
        return True
        
    def stop(self):
        """Stop the audio input handler"""
        logger.info("Stopping audio input handler")
        
        # Signal the thread to stop
        self._stop_event.set()
        self._running = False
        
        # Wait for thread to finish
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=2.0)
            
        # Clean up resources
        if self._is_mic:
            self.close_pyaudio_stream()
            
        # Clear buffers
        if self._data_buffer:
            self._data_buffer.clear()
            
        # Reset variables
        self.chunk_ready = False
        self._thread_running = False
        
        logger.info("Audio input handler stopped")
        
    def pause(self):
        """Pause audio processing"""
        if not self._pause_event.is_set():
            logger.info("Pausing audio input")
            self._pause_event.set()
            self._pause_time = time.time()
            
    def resume(self):
        """Resume audio processing"""
        if self._pause_event.is_set():
            logger.info("Resuming audio input")
            self._pause_event.clear()
            
    def is_paused(self):
        """Check if audio processing is paused"""
        return self._pause_event.is_set()
        
    def is_running(self):
        """Check if the audio input handler is running"""
        return self._thread_running and self._running
        
    def has_audio_finished(self):
        """Check if audio file has finished playing"""
        return self._audio_finished and not self._thread_running
        
    def read_audio_chunk(self):
        """Read an audio chunk from the buffer"""
        if not self._thread_running or len(self._data_buffer) == 0:
            return np.zeros(self.chunk_size, dtype=np.int16)
            
        with self._lock:
            if len(self._data_buffer) > 0:
                chunk = self._data_buffer.popleft()
                self.chunk_ready = len(self._data_buffer) > 0
                return chunk
                
        return np.zeros(self.chunk_size, dtype=np.int16)
    
    def get_audio_data(self):
        """Get audio data from the input source

        Returns
        -------
        numpy.ndarray
            Audio data as a numpy array
        """
        # Get audio data
        if self.input_type == INPUT_MIC:
            # Read from mic
            try:
                y = self.read_audio_chunk()
                if y is None or len(y) == 0:
                    y = np.zeros(self.chunk_size, dtype=np.int16)
            except Exception as e:
                logger.error("Error reading audio chunk: %s", str(e))
                y = np.zeros(self.chunk_size, dtype=np.int16)
        elif self.input_type == INPUT_FILE:
            # Read from file
            y = self.read_audio_chunk()
            if y is None:
                y = np.zeros(self.chunk_size, dtype=np.int16)
        else:
            # No input
            y = np.zeros(self.chunk_size, dtype=np.int16)

        self.current_chunk = y
        return y

    def get_melbank(self, y=None):
        """Calculate the melbank

        Parameters
        ----------
        y : numpy.ndarray, optional
            Audio data, by default None

        Returns
        -------
        numpy.ndarray
            Melbank values
        """
        # This is a placeholder - in a full implementation, 
        # this would calculate the mel spectrogram
        if y is None:
            y = self.current_chunk if self.current_chunk is not None else np.zeros(self.chunk_size)
        
        # Convert to float32
        y_float = y.astype(np.float32) / 32768.0
        
        # Simple FFT implementation for visualization
        N = len(y_float)
        fft_output = np.abs(np.fft.rfft(y_float))
        
        # Simple mel-like scaling (for visualization only)
        num_mel_bands = 24
        fft_len = len(fft_output)
        mel_bands = np.zeros(num_mel_bands)
        
        # Rough approximation of mel scaling
        for i in range(num_mel_bands):
            start_idx = int(fft_len * (i / num_mel_bands)**2)
            end_idx = int(fft_len * ((i+1) / num_mel_bands)**2)
            if end_idx > start_idx:
                mel_bands[i] = np.mean(fft_output[start_idx:end_idx])
        
        # Normalize
        if np.max(mel_bands) > 0:
            mel_bands = mel_bands / np.max(mel_bands)
        
        return mel_bands

    # Compatibility methods for the main script
    
    def get_audio_chunk(self):
        """Compatibility method for the main script"""
        return self.get_audio_data()
        
    def close(self):
        """Compatibility method for the main script"""
        self.stop()
        
    def initialize(self):
        """Compatibility method for the main script"""
        return self.start()
        
    def is_active(self):
        """Compatibility method for the main script"""
        return self.is_running()
        
    def get_frequency_bands(self, num_bands=24):
        """Get frequency bands from audio data
        
        Parameters
        ----------
        num_bands : int, optional
            Number of frequency bands, by default 24
            
        Returns
        -------
        numpy.ndarray
            Frequency bands
        """
        # Use the melbank as frequency bands
        return self.get_melbank()
        
    def get_volume(self):
        """Get current volume level
        
        Returns
        -------
        float
            Volume level between 0 and 1
        """
        if self.current_chunk is None:
            return 0.0
            
        # Calculate RMS volume
        rms = np.sqrt(np.mean(self.current_chunk.astype(np.float32)**2))
        
        # Normalize to 0-1 range
        volume = min(1.0, max(0.0, rms / 32768.0))
        
        return volume

    def is_initialized(self):
        """
        Check if audio input is properly initialized.
        
        Returns:
            bool: True if initialized, False otherwise.
        """
        return self.initialized

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the audio input handler
    config = {
        "SAMPLE_RATE": 48000,
        "CHUNK_SIZE": 1024,
        "CHANNELS": 1,
        "MIC_RATE": 48000,
        "LOOP": True
    }
    
    # Test with mic
    print("Testing with microphone input...")
    audio_handler = AudioInputHandler(config, device_id="mic")
    audio_handler.start()
    
    # Read some chunks
    for i in range(10):
        data = audio_handler.get_audio_data()
        print(f"Chunk {i}: shape={data.shape}, min={np.min(data)}, max={np.max(data)}")
        time.sleep(0.1)
    
    # Stop
    audio_handler.stop()
    print("Microphone test complete")
    
    # Test with file
    if os.path.exists("test.mp3"):
        print("Testing with MP3 file input...")
        audio_handler = AudioInputHandler(config, device_id="test.mp3")
        audio_handler.start()
        
        # Read some chunks
        for i in range(10):
            data = audio_handler.get_audio_data()
            print(f"Chunk {i}: shape={data.shape}, min={np.min(data)}, max={np.max(data)}")
            time.sleep(0.1)
        
        # Stop
        audio_handler.stop()
        print("File test complete")
    else:
        print("No test.mp3 file found for testing") 