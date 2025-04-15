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

# This module handles RGB conversion for the Audio LED Visualization System.
# It takes processed audio data and converts it to RGB values using different methods:
# - Spectrum visualization: Maps frequency bands to colors across the visible spectrum
# - Volume/amplitude visualization: Maps audio volume to brightness or color intensity
# - Beat-reactive visualization: Changes colors or patterns in response to beats
#
# The module is based on the original Psynesthesia code by rho-bit, adapted and
# enhanced for this modular system.

#--------------------------------------
#       IMPORTS
#--------------------------------------

import os
import sys
import logging
import time
import math
import numpy as np
from pathlib import Path

# Configure module logger
logger = logging.getLogger(__name__)

#--------------------------------------
#       CONSTANTS
#--------------------------------------

# Color modes
MODE_SPECTRUM = "spectrum"  # Maps frequencies to colors across the spectrum
MODE_INTENSITY = "intensity"  # Maps volume to brightness/intensity
MODE_CUSTOM = "custom"  # Uses custom color schemes defined in config
MODE_COLORFUL = "colorful" # New mode with colorful visualization

# Default color modes
DEFAULT_COLOR_MODE = MODE_COLORFUL

# Speed of light (m/s)
SPEED_OF_LIGHT = 3e8

# Wavelength ranges (nm)
MIN_WAVELENGTH = 380  # Violet
MAX_WAVELENGTH = 750  # Red

# Color temperature mappings
COLOR_TEMPS = {
    "warm": (255, 147, 41),   # Warm white
    "neutral": (255, 255, 255),  # Neutral white
    "cool": (212, 235, 255)   # Cool white
}

#--------------------------------------
#       FUNCTIONS
#--------------------------------------

def wavelength_to_rgb(wavelength, gamma=0.8, max_intensity=255):
    """
    Convert a wavelength in nm to an RGB color.
    
    This function maps wavelengths in the visible spectrum to RGB colors,
    approximating how humans perceive different wavelengths of light.
    
    Args:
        wavelength (float): Wavelength in nanometers (380-750 nm)
        gamma (float): Gamma correction factor
        max_intensity (int): Maximum intensity value (usually 255)
        
    Returns:
        tuple: RGB color tuple (R, G, B)
    """
    # Clamp wavelength to the visible spectrum
    wavelength = max(MIN_WAVELENGTH, min(MAX_WAVELENGTH, wavelength))
    
    # Calculate RGB values based on wavelength
    if 380 <= wavelength < 440:
        # Violet
        r = (440 - wavelength) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= wavelength < 490:
        # Blue
        r = 0.0
        g = (wavelength - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wavelength < 510:
        # Cyan
        r = 0.0
        g = 1.0
        b = (510 - wavelength) / (510 - 490)
    elif 510 <= wavelength < 580:
        # Green
        r = (wavelength - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= wavelength < 645:
        # Yellow to Orange
        r = 1.0
        g = (645 - wavelength) / (645 - 580)
        b = 0.0
    else:
        # Red
        r = 1.0
        g = 0.0
        b = 0.0
    
    # Apply intensity falloff at spectrum edges
    if wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif wavelength > 700:
        factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 700)
    else:
        factor = 1.0
    
    # Apply gamma correction and scale to max_intensity
    r = int(max_intensity * (r * factor) ** gamma)
    g = int(max_intensity * (g * factor) ** gamma)
    b = int(max_intensity * (b * factor) ** gamma)
    
    return (r, g, b)

def frequency_to_wavelength(frequency):
    """
    Convert a frequency in Hz to a wavelength in nm.
    
    Args:
        frequency (float): Frequency in Hertz
        
    Returns:
        float: Wavelength in nanometers
    """
    # Avoid division by zero
    if frequency <= 0:
        return MIN_WAVELENGTH
    
    # Use a more dynamic mapping to give better color variation
    # Map 20Hz-1kHz to red (620-750nm)
    # Map 1kHz-5kHz to green-yellow (510-580nm)
    # Map 5kHz-20kHz to blue-violet (380-490nm)
    
    if frequency < 1000:  # 20Hz-1kHz (bass)
        # Map to red range (700-620nm)
        normalized = (frequency - 20) / 980
        wavelength = 700 - normalized * 80  # 700nm (deep red) to 620nm (orange-red)
    elif frequency < 5000:  # 1kHz-5kHz (midrange)
        # Map to green-yellow range (580-510nm)
        normalized = (frequency - 1000) / 4000
        wavelength = 580 - normalized * 70  # 580nm (yellow) to 510nm (green)
    else:  # 5kHz-20kHz (high frequencies)
        # Map to blue-violet range (490-380nm)
        normalized = min(1.0, (frequency - 5000) / 15000)
        wavelength = 490 - normalized * 110  # 490nm (cyan) to 380nm (violet)
    
    return max(MIN_WAVELENGTH, min(MAX_WAVELENGTH, wavelength))

def hsv_to_rgb(h, s, v):
    """
    Convert HSV color to RGB.
    
    Args:
        h (float): Hue (0-1)
        s (float): Saturation (0-1)
        v (float): Value (0-1)
        
    Returns:
        tuple: RGB color tuple (R, G, B) with values 0-255
    """
    if s == 0.0:
        # Grayscale
        rgb = (v, v, v)
    else:
        h *= 6.0
        i = int(h)
        f = h - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        if i == 0:
            rgb = (v, t, p)
        elif i == 1:
            rgb = (q, v, p)
        elif i == 2:
            rgb = (p, v, t)
        elif i == 3:
            rgb = (p, q, v)
        elif i == 4:
            rgb = (t, p, v)
        else:
            rgb = (v, p, q)
    
    # Convert to 0-255 range
    return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

def spectrum_to_rgb(spectrum, brightness=1.0):
    """
    Convert a frequency spectrum to an RGB color.
    
    This function takes a spectrum array and calculates an RGB color
    representing the dominant frequencies in the spectrum.
    
    Args:
        spectrum (numpy.ndarray or list): Frequency spectrum array
        brightness (float): Brightness scaling factor (0-1)
        
    Returns:
        tuple: RGB color tuple (R, G, B)
    """
    # Convert to numpy array if it's a list
    if isinstance(spectrum, list):
        spectrum = np.array(spectrum)
        
    if len(spectrum) == 0 or np.max(spectrum) == 0:
        return (0, 0, 0)
    
    # Instead of just using the maximum peak, consider multiple dominant frequencies
    # Get the indices of the top 3 peaks in the spectrum
    num_peaks = min(3, len(spectrum))
    if num_peaks <= 0:
        return (0, 0, 0)
        
    # Sort spectrum and get indices of top peaks
    sorted_indices = np.argsort(spectrum)[::-1][:num_peaks]
    
    # Calculate frequencies for these peaks
    freqs = []
    for idx in sorted_indices:
        # Map the index to a frequency using a logarithmic scale
        # Frequency range: typically 20 Hz - 20 kHz for human hearing
        band_fraction = idx / len(spectrum)
        log_min_freq = math.log10(20)
        log_max_freq = math.log10(20000)
        log_freq = log_min_freq + band_fraction * (log_max_freq - log_min_freq)
        freq = 10 ** log_freq
        freqs.append(freq)
    
    # Convert frequencies to wavelengths
    wavelengths = [frequency_to_wavelength(f) for f in freqs]
    
    # Calculate weight for each peak based on its magnitude relative to the maximum
    peak_values = [spectrum[idx] for idx in sorted_indices]
    total_energy = sum(peak_values)
    if total_energy == 0:
        return (0, 0, 0)
        
    weights = [val / total_energy for val in peak_values]
    
    # Convert wavelengths to RGB and blend them according to weights
    rgb_values = [wavelength_to_rgb(wl) for wl in wavelengths]
    
    # Weighted average of RGB values
    r = 0
    g = 0
    b = 0
    for i in range(num_peaks):
        r += rgb_values[i][0] * weights[i]
        g += rgb_values[i][1] * weights[i]
        b += rgb_values[i][2] * weights[i]
    
    # Apply brightness
    r = min(255, int(r * brightness))
    g = min(255, int(g * brightness))
    b = min(255, int(b * brightness))
    
    return (r, g, b)

def volume_to_rgb(volume, base_color=(255, 255, 255)):
    """
    Convert an audio volume level to an RGB color.
    
    This function maps the volume level to the brightness of the base color.
    
    Args:
        volume (float): Volume level (0-1)
        base_color (tuple): Base RGB color to adjust brightness
        
    Returns:
        tuple: RGB color tuple (R, G, B)
    """
    # Apply volume as brightness
    r = min(255, int(base_color[0] * volume))
    g = min(255, int(base_color[1] * volume))
    b = min(255, int(base_color[2] * volume))
    
    return (r, g, b)

def beat_to_rgb(beat, base_rgb, beat_rgb=None):
    """
    Modify RGB color based on beat detection.
    
    This function returns a different color when a beat is detected.
    
    Args:
        beat (bool): Whether a beat is detected
        base_rgb (tuple): Base RGB color when no beat is detected
        beat_rgb (tuple, optional): RGB color when a beat is detected
        
    Returns:
        tuple: RGB color tuple (R, G, B)
    """
    if beat:
        if beat_rgb is None:
            # Invert or brighten the base color if no beat color specified
            return (
                min(255, base_rgb[0] * 1.5),
                min(255, base_rgb[1] * 1.5),
                min(255, base_rgb[2] * 1.5)
            )
        else:
            return beat_rgb
    else:
        return base_rgb

#--------------------------------------
#       CLASSES
#--------------------------------------

class RGBConverter:
    """
    Convert processed audio data to RGB values.
    
    This class takes processed audio data and converts it to RGB values
    using different visualization methods.
    """
    
    def __init__(self, env_config):
        """
        Initialize the RGB converter.
        
        Args:
            env_config (dict): Environment configuration including visual settings
        """
        self.config = env_config.get('config', {}).get('visual', {})
        
        # Initialize parameters
        self.color_mode = self.config.get('color_mode', DEFAULT_COLOR_MODE)
        self.brightness = self.config.get('brightness', 255) / 255.0  # Normalize to 0-1
        self.custom_colors = self.config.get('custom_colors', [])
        
        # For spectrum mode
        self.output_file = self.config.get('output_file', None)
        self.output_file_handle = None
        
        # Initialize state variables
        self.last_color = (0, 0, 0)
        self.color_time = 0
        self.color_index = 0
        self.hue_offset = 0.0  # For color cycling
        
        # Open output file if specified
        if self.output_file and self.color_mode == MODE_SPECTRUM:
            try:
                self.output_file_handle = open(self.output_file, 'w')
                self.output_file_handle.write("Red\tGreen\tBlue\tTimestamp\n")
                logger.info(f"RGB output file opened: {self.output_file}")
            except Exception as e:
                logger.error(f"Error opening RGB output file: {e}")
                self.output_file_handle = None
        
        logger.info(f"RGB converter initialized (mode={self.color_mode}, brightness={self.brightness:.2f})")
    
    def convert(self, processed_audio):
        """
        Convert processed audio data to RGB values.
        
        Args:
            processed_audio (dict): Processed audio data from AudioProcessor
            
        Returns:
            tuple: RGB color tuple (R, G, B) with integer values
        """
        if processed_audio is None:
            return (0, 0, 0)
        
        # Extract data from processed audio
        spectrum = processed_audio.get('spectrum', [])
        volume = processed_audio.get('normalized_volume', 0)
        beat = processed_audio.get('beat', False)
        
        # Convert based on selected color mode
        if self.color_mode == MODE_SPECTRUM:
            # Directly use the spectrum-to-RGB mapping
            rgb = self.spectrum_to_rgb(spectrum)
        elif self.color_mode == MODE_INTENSITY:
            # Map volume to brightness with HSV
            hue = (self.hue_offset * 0.1) % 1.0  # Slowly cycle through hues
            saturation = 1.0  # Full saturation
            value = min(1.0, volume * 1.2) * self.brightness  # Map volume to brightness
            rgb = hsv_to_rgb(hue, saturation, value)
            
            # Update hue offset
            self.hue_offset += 0.05
        elif self.color_mode == MODE_CUSTOM:
            rgb = self._custom_mode(spectrum, volume, beat)
        else:
            # Default to a colorful visualization
            rgb = self._colorful_visualization(spectrum, volume, beat)
        
        # Apply beat effect
        if beat:
            rgb = beat_to_rgb(beat, rgb)
        
        # Write to output file if enabled
        if self.output_file_handle:
            try:
                self.output_file_handle.write(f"{rgb[0]}\t{rgb[1]}\t{rgb[2]}\t{time.time()}\n")
                self.output_file_handle.flush()
            except Exception as e:
                logger.error(f"Error writing to RGB output file: {e}")
        
        # Update state
        self.last_color = rgb
        
        # Ensure integers are returned for pygame compatibility
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def spectrum_to_rgb(self, spectrum):
        """
        Convert a spectrum to RGB color.
        
        This is a convenience method that delegates to the global spectrum_to_rgb function,
        applying the configured brightness. Ensures values are integers for pygame.
        
        Args:
            spectrum (numpy.ndarray): Frequency spectrum
            
        Returns:
            tuple: RGB color tuple (R, G, B) with integer values
        """
        rgb = spectrum_to_rgb(spectrum, self.brightness)
        # Make sure we're returning integers (pygame requires this)
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def _colorful_visualization(self, spectrum, volume, beat):
        """
        Create a colorful visualization that cycles through the color wheel.
        
        Args:
            spectrum (numpy.ndarray): Frequency spectrum
            volume (float): Normalized volume level (0-1)
            beat (bool): Whether a beat is detected
            
        Returns:
            tuple: RGB color tuple (R, G, B) with integer values
        """
        # Calculate a dominant frequency index (0.0 - 1.0)
        if len(spectrum) > 0 and np.max(spectrum) > 0:
            # Weight more towards bass (first third of the spectrum)
            bass_weight = 0.6
            mid_weight = 0.3
            high_weight = 0.1
            
            bass_idx = int(len(spectrum) * 0.33)
            mid_idx = int(len(spectrum) * 0.67)
            
            bass_energy = np.sum(spectrum[:bass_idx]) * bass_weight if bass_idx > 0 else 0
            mid_energy = np.sum(spectrum[bass_idx:mid_idx]) * mid_weight if mid_idx > bass_idx else 0
            high_energy = np.sum(spectrum[mid_idx:]) * high_weight if mid_idx < len(spectrum) else 0
            
            # Calculate the frequency center based on weighted energy
            total_energy = bass_energy + mid_energy + high_energy
            if total_energy > 0:
                freq_factor = (bass_energy * 0.2 + mid_energy * 0.5 + high_energy * 0.9) / total_energy
            else:
                freq_factor = 0.5
        else:
            freq_factor = 0.5
        
        # Use audio data to influence the hue
        # Hue is determined by a combination of:
        # 1. Slow automatic cycle (hue_offset)
        # 2. Frequency content (freq_factor)
        # 3. Beat detection (adds a small jump)
        hue_base = (self.hue_offset * 0.02) % 1.0  # Slow automatic cycle
        hue_freq = (freq_factor * 0.8) % 1.0       # Frequency influence
        hue_beat = 0.1 if beat else 0.0            # Beat influence
        
        # Calculate final hue (0-1)
        hue = (hue_base + hue_freq + hue_beat) % 1.0
        
        # Saturation slightly reduced during quiet parts
        saturation = 0.8 + volume * 0.2
        
        # Value (brightness) based on volume with a minimum
        value = min(1.0, 0.3 + volume * 0.7) * self.brightness
        
        # Convert HSV to RGB
        rgb = hsv_to_rgb(hue, saturation, value)
        
        # Update state for next frame
        self.hue_offset += 0.1
        
        # Ensure integers are returned
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def _custom_mode(self, spectrum, volume, beat):
        """
        Generate RGB values using custom color schemes.
        
        This is a more flexible method that can be customized based on
        specific preferences or config settings.
        
        Args:
            spectrum (numpy.ndarray): Frequency spectrum
            volume (float): Normalized volume level (0-1)
            beat (bool): Whether a beat is detected
            
        Returns:
            tuple: RGB color tuple (R, G, B) with integer values
        """
        # If no custom colors defined, use HSV color wheel
        if not self.custom_colors:
            # Use time to cycle through colors
            now = time.time()
            hue = (now * 0.1) % 1.0
            
            # Adjust saturation based on beat
            saturation = 1.0 if beat else 0.8
            
            # Adjust value (brightness) based on volume
            value = min(1.0, 0.3 + volume * 0.7) * self.brightness
            
            rgb = hsv_to_rgb(hue, saturation, value)
            return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        
        # Use custom colors if available
        elif len(self.custom_colors) > 0:
            # If beat detected, change color
            if beat:
                self.color_index = (self.color_index + 1) % len(self.custom_colors)
            
            # Get the current color
            color = self.custom_colors[self.color_index]
            
            # Adjust brightness based on volume
            adjusted_brightness = min(1.0, 0.3 + volume * 0.7) * self.brightness
            
            return (
                int(min(255, color[0] * adjusted_brightness)),
                int(min(255, color[1] * adjusted_brightness)),
                int(min(255, color[2] * adjusted_brightness))
            )
        
        # Fallback
        rgb = volume_to_rgb(volume * self.brightness)
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def close(self):
        """
        Close the RGB converter and clean up resources.
        """
        # Close output file if open
        if self.output_file_handle:
            try:
                self.output_file_handle.close()
                logger.info("RGB output file closed")
            except Exception as e:
                logger.error(f"Error closing RGB output file: {e}")
            
            self.output_file_handle = None

# Test the module if run directly
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import rgb2hex
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test wavelength to RGB conversion
    print("Testing wavelength to RGB conversion:")
    wavelengths = [380, 450, 500, 550, 600, 650, 700, 750]
    colors = [wavelength_to_rgb(wl) for wl in wavelengths]
    
    for wl, color in zip(wavelengths, colors):
        print(f"  {wl} nm -> RGB {color}")
    
    # Test frequency to RGB conversion with a synthetic spectrum
    print("\nTesting frequency to RGB conversion:")
    
    # Create a synthetic frequency spectrum
    bands = 64
    test_spectrums = [
        np.zeros(bands),                            # Empty spectrum
        np.ones(bands),                             # Flat spectrum
        np.hstack([np.ones(bands//4), np.zeros(bands - bands//4)]),  # Low frequencies
        np.hstack([np.zeros(bands//2), np.ones(bands//4), np.zeros(bands//4)]),  # Mid frequencies
        np.hstack([np.zeros(3*bands//4), np.ones(bands//4)])         # High frequencies
    ]
    
    spectrum_names = ["Empty", "Flat", "Bass", "Mid", "Treble"]
    
    # Plot the spectrum and resulting colors
    plt.figure(figsize=(12, 6))
    
    for i, (spectrum, name) in enumerate(zip(test_spectrums, spectrum_names)):
        # Calculate RGB color
        rgb = spectrum_to_rgb(spectrum)
        
        # Plot spectrum
        plt.subplot(len(test_spectrums), 2, i*2 + 1)
        plt.bar(range(len(spectrum)), spectrum)
        plt.title(f"{name} Spectrum")
        plt.ylim(0, 1.1)
        
        # Plot color
        plt.subplot(len(test_spectrums), 2, i*2 + 2)
        plt.fill([0, 1, 1, 0], [0, 0, 1, 1], color=[c/255 for c in rgb])
        plt.title(f"RGB: {rgb}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show() 