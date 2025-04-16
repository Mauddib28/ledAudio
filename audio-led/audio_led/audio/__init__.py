"""
Audio module for LED Audio Visualization System.

This module handles audio input from various sources such as files, microphones,
and line-in devices, providing a consistent interface for audio data retrieval.
"""

from .input_handler import (
    AudioInputHandler,
    FileAudioInput,
    MicrophoneAudioInput,
    LineInAudioInput,
    InputHandlerFactory,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_CHANNELS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_BUFFER_SIZE
)

__all__ = [
    'AudioInputHandler',
    'FileAudioInput',
    'MicrophoneAudioInput',
    'LineInAudioInput',
    'InputHandlerFactory',
    'DEFAULT_SAMPLE_RATE',
    'DEFAULT_CHANNELS',
    'DEFAULT_CHUNK_SIZE',
    'DEFAULT_BUFFER_SIZE'
] 