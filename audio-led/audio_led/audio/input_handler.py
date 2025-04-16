import os
import time
import wave
import logging
import threading
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

# Try to import optional dependencies
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    logging.warning("PyAudio not available - install with: pip install pyaudio")

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    logging.warning("PyDub not available - install with: pip install pydub")

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    logging.warning("Soundfile not available - install with: pip install soundfile")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logging.warning("Librosa not available - install with: pip install librosa")

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    logging.warning("Sounddevice not available - install with: pip install sounddevice")

# Constants
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 2
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_BUFFER_SIZE = 10  # Number of chunks to buffer

logger = logging.getLogger(__name__)

class AudioInputHandler(ABC):
    """Base class for audio input handlers."""
    
    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, channels=DEFAULT_CHANNELS, 
                 chunk_size=DEFAULT_CHUNK_SIZE, buffer_size=DEFAULT_BUFFER_SIZE):
        """Initialize the audio input handler.
        
        Args:
            sample_rate (int): Sample rate in Hz
            channels (int): Number of audio channels
            chunk_size (int): Size of audio chunks in frames
            buffer_size (int): Size of the buffer in chunks
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.running = False
        self.initialized = False
        
        # Buffer for audio data
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.RLock()
        
        # Thread for processing audio
        self.thread = None
        self.thread_exception = None
        
        # Volume tracking
        self.current_volume = 0.0
        
        # Audio properties
        self.duration = 0
        self.position = 0
        self.start_time = 0

    def initialize(self):
        """Initialize the audio input handler."""
        if self.initialized:
            return True
            
        try:
            self._setup_input()
            self.initialized = True
            logger.info(f"Audio input handler initialized (rate={self.sample_rate}, channels={self.channels}, chunk={self.chunk_size})")
            return True
        except Exception as e:
            logger.error(f"Error initializing audio input handler: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start(self):
        """Start the audio input handler."""
        if self.running:
            logger.warning("Audio input handler already running")
            return True
            
        if not self.initialized and not self.initialize():
            logger.error("Failed to initialize audio input handler")
            return False
            
        try:
            self.running = True
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._input_thread)
            self.thread.daemon = True
            self.thread.start()
            logger.info("Audio input handler started")
            return True
        except Exception as e:
            logger.error(f"Error starting audio input handler: {e}")
            self.running = False
            return False

    def stop(self):
        """Stop the audio input handler."""
        if not self.running:
            logger.info("Audio input handler already stopped")
            return True
            
        self.running = False
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            logger.info("Waiting for audio input thread to finish...")
            self.thread.join(timeout=2.0)
            runtime = time.time() - self.start_time
            logger.info(f"Audio input thread finished. Runtime: {runtime:.2f} seconds")
            
        # Clear buffer
        with self.buffer_lock:
            self.buffer.clear()
            
        return True

    def close(self):
        """Close the audio input handler and release resources."""
        if self.running:
            self.stop()
            
        if self.initialized:
            try:
                self._cleanup()
                logger.info("Audio input handler closed")
            except Exception as e:
                logger.error(f"Error closing audio input handler: {e}")
                
        self.initialized = False
        return True

    def get_audio_chunk(self):
        """Get a chunk of audio data.
        
        Returns:
            numpy.ndarray: Audio data or None if no data is available
        """
        if not self.running:
            return None
            
        if self.thread_exception:
            logger.error(f"Audio input thread exception: {self.thread_exception}")
            self.thread_exception = None
            return None
            
        with self.buffer_lock:
            if len(self.buffer) > 0:
                return self.buffer.popleft()
            else:
                return None

    def is_active(self):
        """Check if the audio input handler is active.
        
        Returns:
            bool: True if the handler is active, False otherwise
        """
        return self.running and not self.has_audio_finished()

    def is_initialized(self):
        """Check if the audio input handler is initialized.
        
        Returns:
            bool: True if the handler is initialized, False otherwise
        """
        return self.initialized

    def has_audio_finished(self):
        """Check if the audio has finished.
        
        Returns:
            bool: True if the audio has finished, False otherwise
        """
        # Base implementation for continuous sources (mic, line-in)
        # File-based sources will override this
        return False

    def get_volume(self):
        """Get the current audio volume.
        
        Returns:
            float: Current volume level (0.0 to 1.0)
        """
        return self.current_volume

    def get_sample_rate(self):
        """Get the sample rate.
        
        Returns:
            int: Sample rate in Hz
        """
        return self.sample_rate

    def get_channels(self):
        """Get the number of channels.
        
        Returns:
            int: Number of channels
        """
        return self.channels

    def get_chunk_size(self):
        """Get the chunk size.
        
        Returns:
            int: Chunk size in frames
        """
        return self.chunk_size

    def get_progress(self):
        """Get the current playback progress.
        
        Returns:
            float: Current progress (0.0 to 1.0)
        """
        if self.duration <= 0:
            return 0.0
        return min(1.0, self.position / self.duration)

    def _calculate_volume(self, data_array):
        """Calculate the volume of an audio chunk.
        
        Args:
            data_array (numpy.ndarray): Audio data
            
        Returns:
            float: Volume level (0.0 to 1.0)
        """
        if data_array is None or len(data_array) == 0:
            return 0.0
            
        try:
            # Ensure we're working with a clean array
            if not np.all(np.isfinite(data_array)):
                data_array = np.nan_to_num(data_array)
                
            # Calculate RMS (root mean square) volume
            # For multichannel audio, average across channels
            if len(data_array.shape) > 1 and data_array.shape[1] > 1:
                # Multi-channel data
                squared = np.square(data_array)
                if not np.all(np.isfinite(squared)):
                    squared = np.nan_to_num(squared)
                    
                mean_squares = np.mean(squared)
                if not np.isfinite(mean_squares) or mean_squares < 0:
                    return 0.0
                    
                rms = np.sqrt(mean_squares)
            else:
                # Mono data
                squared = np.square(data_array)
                if not np.all(np.isfinite(squared)):
                    squared = np.nan_to_num(squared)
                    
                mean_squares = np.mean(squared)
                if not np.isfinite(mean_squares) or mean_squares < 0:
                    return 0.0
                    
                rms = np.sqrt(mean_squares)
                
            # Normalize to 0.0-1.0 range (assuming audio is normalized to -1.0 to 1.0)
            # Apply some non-linear scaling to make it more responsive
            volume = np.power(min(1.0, rms), 0.5)
            
            # Check for NaN or Inf values
            if not np.isfinite(volume):
                return 0.0
                
            return float(volume)
        except Exception as e:
            logger.error(f"Error calculating volume: {e}")
            return 0.0

    @abstractmethod
    def _setup_input(self):
        """Set up the audio input source."""
        pass

    @abstractmethod
    def _input_thread(self):
        """Thread for processing audio input."""
        pass

    @abstractmethod
    def _cleanup(self):
        """Clean up resources."""
        pass


class FileAudioInput(AudioInputHandler):
    """Audio input handler for audio files."""
    
    def __init__(self, file_path, sample_rate=None, channels=None, chunk_size=DEFAULT_CHUNK_SIZE,
                 buffer_size=DEFAULT_BUFFER_SIZE):
        """Initialize the file audio input handler.
        
        Args:
            file_path (str): Path to the audio file
            sample_rate (int, optional): Sample rate to use (None for file's native rate)
            channels (int, optional): Number of channels to use (None for file's native channels)
            chunk_size (int): Size of audio chunks in frames
            buffer_size (int): Size of the buffer in chunks
        """
        # Use default values initially, will be updated from file
        super().__init__(
            sample_rate=sample_rate or DEFAULT_SAMPLE_RATE,
            channels=channels or DEFAULT_CHANNELS,
            chunk_size=chunk_size,
            buffer_size=buffer_size
        )
        
        self.file_path = file_path
        self.requested_sample_rate = sample_rate
        self.requested_channels = channels
        
        # Audio data
        self.audio_data = None
        self.file_sample_rate = None
        self.file_channels = None
        self.format = None
        self.current_frame = 0
        self.total_frames = 0
        self.finished = False
        
        # File format detection
        self.file_exists = False
        if self.file_path:
            self.file_exists = os.path.isfile(self.file_path)
            if not self.file_exists:
                logger.error(f"File not found: {file_path}")
            else:
                self.format = self._detect_format(file_path)
                logger.info(f"Detected audio format: {self.format}")

    def has_audio_finished(self):
        """Check if the audio file has finished playing.
        
        Returns:
            bool: True if the audio has finished, False otherwise
        """
        return self.finished

    def _detect_format(self, file_path):
        """Detect the format of an audio file.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            str: Audio format (wav, mp3, etc.) or None if unknown
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        # Handle case-insensitive extensions
        if ext in ['.wav', '.wave']:
            return 'wav'
        elif ext in ['.mp3']:
            return 'mp3'
        elif ext in ['.ogg', '.oga']:
            return 'ogg'
        elif ext in ['.flac']:
            return 'flac'
        elif ext in ['.m4a', '.aac']:
            return 'aac'
        else:
            logger.warning(f"Unknown audio format: {ext}")
            return None

    def _setup_input(self):
        """Set up the audio file input."""
        if not self.file_exists:
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        if not self.format:
            raise ValueError(f"Unsupported audio format: {self.file_path}")
            
        logger.info(f"Loading audio file: {os.path.basename(self.file_path)}")
        
        # Choose the best available loader based on format and available libraries
        if self.format == 'wav' and HAS_SOUNDFILE:
            self._load_with_soundfile()
        elif HAS_LIBROSA:
            self._load_with_librosa()
        elif HAS_SOUNDFILE:
            self._load_with_soundfile()
        elif HAS_PYDUB:
            self._load_with_pydub()
        else:
            raise ImportError("No suitable audio library found. Install soundfile, librosa, or pydub.")
            
        # Update duration and channel count
        self.total_frames = len(self.audio_data)
        self.duration = self.total_frames / self.file_sample_rate
        
        # Use requested sample rate and channels if provided, otherwise use file's native values
        self.sample_rate = self.requested_sample_rate or self.file_sample_rate
        self.channels = self.requested_channels or self.file_channels
        
        logger.info(f"Audio file loaded: {self.total_frames} samples, {self.file_sample_rate} Hz")
        
        return True

    def _load_with_soundfile(self):
        """Load audio file using soundfile."""
        try:
            audio_data, self.file_sample_rate = sf.read(self.file_path, dtype='float32')
            
            # Handle mono/stereo conversion
            if len(audio_data.shape) == 1:  # Mono
                self.file_channels = 1
                # Convert to 2D array for consistent handling
                audio_data = audio_data.reshape(-1, 1)
            else:  # Multi-channel
                self.file_channels = audio_data.shape[1]
                
            # Convert to float32 and normalize to -1.0 to 1.0
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
                
            self.audio_data = audio_data
            logger.info(f"Loading {os.path.basename(self.file_path)} with soundfile")
            
        except Exception as e:
            logger.error(f"Error loading with soundfile: {e}")
            raise

    def _load_with_librosa(self):
        """Load audio file using librosa."""
        try:
            audio_data, self.file_sample_rate = librosa.load(
                self.file_path, 
                sr=self.requested_sample_rate or None,  # Use None for native sample rate
                mono=False  # Keep all channels
            )
            
            # Librosa loads audio as mono by default if mono=True, or as (n_channels, n_samples) if mono=False
            if len(audio_data.shape) == 1:  # Mono
                self.file_channels = 1
                # Convert to (n_samples, n_channels) format
                audio_data = audio_data.reshape(-1, 1)
            else:  # Multi-channel, shape is (n_channels, n_samples)
                self.file_channels = audio_data.shape[0]
                # Transpose to (n_samples, n_channels) format
                audio_data = audio_data.T
                
            # Already normalized by librosa
            self.audio_data = audio_data
            logger.info(f"Loading {os.path.basename(self.file_path)} with librosa")
            
        except Exception as e:
            logger.error(f"Error loading with librosa: {e}")
            raise

    def _load_with_pydub(self):
        """Load audio file using pydub."""
        try:
            audio_segment = AudioSegment.from_file(self.file_path)
            
            # Get audio properties
            self.file_sample_rate = audio_segment.frame_rate
            self.file_channels = audio_segment.channels
            
            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            
            # Reshape for multi-channel audio
            if self.file_channels > 1:
                samples = samples.reshape((-1, self.file_channels))
                
            # Convert to float32 and normalize to -1.0 to 1.0
            if samples.dtype != np.float32:
                # Get max value based on sample width
                max_value = float(1 << (8 * audio_segment.sample_width - 1))
                samples = samples.astype(np.float32) / max_value
                
            self.audio_data = samples
            logger.info(f"Loading {os.path.basename(self.file_path)} with pydub")
            
        except Exception as e:
            logger.error(f"Error loading with pydub: {e}")
            raise

    def _input_thread(self):
        """Process audio file data in a separate thread."""
        try:
            # Initialize position tracking
            self.position = 0
            self.current_frame = 0
            self.finished = False
            last_time = time.time()
            
            # Process until stopped or end of file
            while self.running and self.current_frame < self.total_frames:
                # Calculate the next chunk
                end_frame = min(self.current_frame + self.chunk_size, self.total_frames)
                chunk = self.audio_data[self.current_frame:end_frame]
                
                # Perform resampling if needed
                if self.sample_rate != self.file_sample_rate:
                    # Simple resampling by repeating or skipping samples
                    # For production, use a proper resampling library like librosa or scipy
                    rate_ratio = self.sample_rate / self.file_sample_rate
                    new_len = int(len(chunk) * rate_ratio)
                    indices = np.linspace(0, len(chunk) - 1, new_len)
                    chunk = np.array([chunk[int(i)] for i in indices])
                
                # Channel conversion if needed
                if self.channels != self.file_channels:
                    if self.file_channels == 1 and self.channels == 2:
                        # Mono to stereo: duplicate the channel
                        chunk = np.column_stack((chunk, chunk))
                    elif self.file_channels == 2 and self.channels == 1:
                        # Stereo to mono: average the channels
                        chunk = np.mean(chunk, axis=1, keepdims=True)
                    else:
                        # More complex channel conversions not implemented
                        logger.warning(f"Unsupported channel conversion: {self.file_channels} to {self.channels}")
                
                # Calculate volume
                self.current_volume = self._calculate_volume(chunk)
                
                # Add to buffer
                with self.buffer_lock:
                    self.buffer.append(chunk)
                
                # Update position
                self.current_frame = end_frame
                current_time = time.time()
                self.position = self.current_frame / self.file_sample_rate
                
                # Sleep to simulate real-time processing and prevent buffer overflow
                buffer_fullness = len(self.buffer) / self.buffer.maxlen
                if buffer_fullness > 0.8:
                    # Buffer getting full, slow down
                    time.sleep(0.01)
                
                # Check if we've reached the end
                if self.current_frame >= self.total_frames:
                    logger.info("End of audio file reached")
                    self.finished = True
                    break
                
            # Mark as finished when thread exits
            self.finished = True
            
        except Exception as e:
            logger.error(f"Error in file input thread: {e}")
            import traceback
            traceback.print_exc()
            self.thread_exception = e
            self.finished = True

    def _cleanup(self):
        """Clean up resources."""
        # Release memory
        self.audio_data = None


class MicrophoneAudioInput(AudioInputHandler):
    """Audio input handler for microphone input."""
    
    def __init__(self, device_id=None, sample_rate=DEFAULT_SAMPLE_RATE, channels=DEFAULT_CHANNELS,
                 chunk_size=DEFAULT_CHUNK_SIZE, buffer_size=DEFAULT_BUFFER_SIZE):
        """Initialize the microphone audio input handler.
        
        Args:
            device_id: Device ID or name for the microphone (None for default)
            sample_rate (int): Sample rate in Hz
            channels (int): Number of channels
            chunk_size (int): Size of audio chunks in frames
            buffer_size (int): Size of the buffer in chunks
        """
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            chunk_size=chunk_size,
            buffer_size=buffer_size
        )
        
        self.device_id = device_id
        
        # PyAudio resources
        self.pyaudio = None
        self.stream = None
        
        # Check if PyAudio is available
        if not HAS_PYAUDIO and not HAS_SOUNDDEVICE:
            logger.error("Neither PyAudio nor SoundDevice available. Cannot use microphone input.")
        
    def _setup_input(self):
        """Set up the microphone input."""
        if HAS_SOUNDDEVICE:
            return self._setup_sounddevice()
        elif HAS_PYAUDIO:
            return self._setup_pyaudio()
        else:
            raise ImportError("No suitable audio library found. Install PyAudio or SoundDevice.")
    
    def _setup_sounddevice(self):
        """Set up using sounddevice."""
        try:
            # List available devices
            devices = sd.query_devices()
            logger.info(f"Found {len(devices)} audio devices")
            
            # Find the device
            device_id = self.device_id
            if device_id is None:
                device_id = sd.default.device[0]  # Default input device
                logger.info(f"Using default input device: {device_id}")
            
            # Get device info
            try:
                device_info = sd.query_devices(device_id, 'input')
                logger.info(f"Using input device: {device_info['name']}")
                
                # Update parameters based on device capabilities
                if self.sample_rate not in device_info['default_samplerate']:
                    logger.warning(f"Requested sample rate {self.sample_rate} not supported by device")
                    self.sample_rate = int(device_info['default_samplerate'])
                    logger.info(f"Using device sample rate: {self.sample_rate}")
                
                max_channels = device_info['max_input_channels']
                if self.channels > max_channels:
                    logger.warning(f"Requested {self.channels} channels, but device only supports {max_channels}")
                    self.channels = max_channels
                
                return True
            except Exception as e:
                logger.error(f"Error setting up sounddevice: {e}")
                return False
        except Exception as e:
            logger.error(f"Error setting up sounddevice: {e}")
            return False
    
    def _setup_pyaudio(self):
        """Set up using PyAudio."""
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            # List available devices
            device_count = self.pyaudio.get_device_count()
            logger.info(f"Found {device_count} audio devices")
            
            # Find the device
            device_index = None
            if self.device_id is not None:
                # Try to find the device by ID or name
                if isinstance(self.device_id, int):
                    device_index = self.device_id
                else:
                    # Search by name
                    for i in range(device_count):
                        device_info = self.pyaudio.get_device_info_by_index(i)
                        if self.device_id.lower() in device_info['name'].lower():
                            device_index = i
                            break
            
            # If no device found, use default
            if device_index is None:
                device_index = self.pyaudio.get_default_input_device_info()['index']
                logger.info(f"Using default input device: {device_index}")
            
            # Get device info
            try:
                device_info = self.pyaudio.get_device_info_by_index(device_index)
                logger.info(f"Using input device: {device_info['name']}")
                
                # Check if the device supports the requested format
                max_channels = int(device_info['maxInputChannels'])
                if self.channels > max_channels:
                    logger.warning(f"Requested {self.channels} channels, but device only supports {max_channels}")
                    self.channels = max_channels
                
                # Open the stream
                self.stream = self.pyaudio.open(
                    format=pyaudio.paFloat32,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.chunk_size
                )
                
                logger.info(f"Microphone input initialized: {self.sample_rate} Hz, {self.channels} channels")
                return True
            except Exception as e:
                logger.error(f"Error setting up PyAudio: {e}")
                if self.pyaudio:
                    self.pyaudio.terminate()
                    self.pyaudio = None
                return False
        except Exception as e:
            logger.error(f"Error setting up PyAudio: {e}")
            return False
    
    def _input_thread(self):
        """Read audio data in a separate thread."""
        try:
            if HAS_SOUNDDEVICE:
                self._sounddevice_thread()
            elif HAS_PYAUDIO and self.stream:
                self._pyaudio_thread()
            else:
                raise RuntimeError("No audio input method available")
        except Exception as e:
            logger.error(f"Error in microphone input thread: {e}")
            import traceback
            traceback.print_exc()
            self.thread_exception = e
    
    def _sounddevice_thread(self):
        """Thread for processing audio input using sounddevice."""
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Sounddevice status: {status}")
            
            # Convert to float32 if needed
            if indata.dtype != np.float32:
                data = indata.astype(np.float32)
            else:
                data = indata.copy()
            
            # Calculate volume
            self.current_volume = self._calculate_volume(data)
            
            # Add to buffer
            with self.buffer_lock:
                self.buffer.append(data)
        
        try:
            # Start the stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=self.channels,
                dtype='float32',
                callback=callback
            ):
                # Just wait until stopped
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in sounddevice thread: {e}")
            raise
    
    def _pyaudio_thread(self):
        """Thread for processing audio input using PyAudio."""
        try:
            # Process until stopped
            while self.running:
                try:
                    # Read audio data
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Convert to numpy array
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    
                    # Reshape for multi-channel
                    if self.channels > 1:
                        audio_data = audio_data.reshape(-1, self.channels)
                    else:
                        audio_data = audio_data.reshape(-1, 1)
                    
                    # Calculate volume
                    self.current_volume = self._calculate_volume(audio_data)
                    
                    # Add to buffer
                    with self.buffer_lock:
                        self.buffer.append(audio_data)
                    
                    # Sleep a bit to prevent high CPU usage
                    time.sleep(0.001)
                    
                except Exception as e:
                    if self.running:  # Only log errors if still running
                        logger.error(f"Error reading from microphone: {e}")
                        time.sleep(0.1)  # Wait before retrying
            
        except Exception as e:
            logger.error(f"Error in PyAudio thread: {e}")
            raise
    
    def _cleanup(self):
        """Clean up resources."""
        try:
            if HAS_PYAUDIO and self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None
                
            logger.info("Microphone resources released")
        except Exception as e:
            logger.error(f"Error closing microphone: {e}")
            
    @staticmethod
    def list_devices():
        """List available audio input devices.
        
        Returns:
            list: List of available devices
        """
        devices = []
        
        if HAS_SOUNDDEVICE:
            try:
                sd_devices = sd.query_devices()
                for i, device in enumerate(sd_devices):
                    if device['max_input_channels'] > 0:
                        devices.append({
                            'index': i,
                            'name': device['name'],
                            'channels': device['max_input_channels'],
                            'default': i == sd.default.device[0],
                            'sample_rates': [int(device['default_samplerate'])],
                            'api': 'sounddevice'
                        })
            except Exception as e:
                logger.error(f"Error listing sounddevice devices: {e}")
        
        if HAS_PYAUDIO:
            try:
                p = pyaudio.PyAudio()
                device_count = p.get_device_count()
                default_device = p.get_default_input_device_info()['index']
                
                for i in range(device_count):
                    try:
                        device_info = p.get_device_info_by_index(i)
                        if device_info['maxInputChannels'] > 0:
                            devices.append({
                                'index': i,
                                'name': device_info['name'],
                                'channels': device_info['maxInputChannels'],
                                'default': i == default_device,
                                'sample_rates': [int(device_info['defaultSampleRate'])],
                                'api': 'pyaudio'
                            })
                    except Exception:
                        pass  # Skip devices with errors
                
                p.terminate()
            except Exception as e:
                logger.error(f"Error listing PyAudio devices: {e}")
        
        return devices 

class LineInAudioInput(MicrophoneAudioInput):
    """Audio input handler for line-in input.
    
    This is basically the same as MicrophoneAudioInput but specifically
    targets line-in devices and may have different default settings.
    """
    
    def __init__(self, device_id=None, sample_rate=DEFAULT_SAMPLE_RATE, channels=DEFAULT_CHANNELS,
                 chunk_size=DEFAULT_CHUNK_SIZE, buffer_size=DEFAULT_BUFFER_SIZE):
        """Initialize the line-in audio input handler.
        
        Args:
            device_id: Device ID or name for the line-in (None for default)
            sample_rate (int): Sample rate in Hz
            channels (int): Number of channels
            chunk_size (int): Size of audio chunks in frames
            buffer_size (int): Size of the buffer in chunks
        """
        super().__init__(
            device_id=device_id,
            sample_rate=sample_rate,
            channels=channels,
            chunk_size=chunk_size,
            buffer_size=buffer_size
        )


class InputHandlerFactory:
    """Factory for creating audio input handlers."""
    
    @staticmethod
    def create_handler(input_type, device_id=None, sample_rate=None, channels=None, 
                       chunk_size=DEFAULT_CHUNK_SIZE, buffer_size=DEFAULT_BUFFER_SIZE):
        """Create an audio input handler.
        
        Args:
            input_type (str): Type of input ('file', 'microphone', 'line-in')
            device_id: Device ID, path or name
            sample_rate (int, optional): Sample rate in Hz
            channels (int, optional): Number of channels
            chunk_size (int): Size of audio chunks in frames
            buffer_size (int): Size of the buffer in chunks
            
        Returns:
            AudioInputHandler: Appropriate input handler
        """
        logger.info(f"Setting up input type for device ID: {device_id}")
        
        if input_type == 'file':
            handler = FileAudioInput(
                file_path=device_id,
                sample_rate=sample_rate,
                channels=channels,
                chunk_size=chunk_size,
                buffer_size=buffer_size
            )
            logger.info(f"Input set to file: {os.path.basename(device_id)}")
            
        elif input_type == 'microphone':
            handler = MicrophoneAudioInput(
                device_id=device_id,
                sample_rate=sample_rate or DEFAULT_SAMPLE_RATE,
                channels=channels or DEFAULT_CHANNELS,
                chunk_size=chunk_size,
                buffer_size=buffer_size
            )
            logger.info(f"Input set to microphone: {device_id or 'default'}")
            
        elif input_type == 'line-in':
            handler = LineInAudioInput(
                device_id=device_id,
                sample_rate=sample_rate or DEFAULT_SAMPLE_RATE,
                channels=channels or DEFAULT_CHANNELS,
                chunk_size=chunk_size,
                buffer_size=buffer_size
            )
            logger.info(f"Input set to line-in: {device_id or 'default'}")
            
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
            
        return handler
    
    @staticmethod
    def get_available_inputs():
        """Get available audio input sources.
        
        Returns:
            dict: Available inputs by type
        """
        inputs = {
            'file': [],
            'microphone': [],
            'line-in': []
        }
        
        # Get microphone and line-in devices
        devices = MicrophoneAudioInput.list_devices()
        
        for device in devices:
            # Add to appropriate category
            if 'line' in device['name'].lower() or 'input' in device['name'].lower():
                inputs['line-in'].append(device)
            else:
                inputs['microphone'].append(device)
        
        return inputs 