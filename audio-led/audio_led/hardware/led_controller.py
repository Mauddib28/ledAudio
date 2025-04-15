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

# This module provides LED control functionality for the Audio LED Visualization System.
# It handles:
# - Control of different LED strip types (WS2812B, APA102, etc.)
# - Support for different hardware interfaces (GPIO, SPI, I2C)
# - Virtual LED output for testing and development
# - Hardware abstraction layer for different platforms
#
# The LED controller accepts visualization data and maps it to
# the physical LED layout.

#--------------------------------------
#       IMPORTS
#--------------------------------------

import time
import queue
import logging
import threading
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

# Configure module logger
logger = logging.getLogger(__name__)

# Optional imports based on environment
try:
    import board
    import neopixel
    HAS_NEOPIXEL = True
except ImportError:
    HAS_NEOPIXEL = False
    logger.warning("NeoPixel library not available")

try:
    import board
    import adafruit_dotstar
    HAS_DOTSTAR = True
except ImportError:
    HAS_DOTSTAR = False
    logger.warning("DotStar library not available")

try:
    import RPi.GPIO as GPIO
    HAS_RPI_GPIO = True
except ImportError:
    HAS_RPI_GPIO = False
    logger.warning("RPi.GPIO library not available")

try:
    from rpi_ws281x import PixelStrip, Color
    HAS_RPI_WS281X = True
except ImportError:
    HAS_RPI_WS281X = False
    logger.warning("rpi_ws281x library not available")

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    logger.warning("Pygame not available, virtual display will be limited")

#--------------------------------------
#       CONSTANTS
#--------------------------------------

# LED strip types
class LEDStripType(Enum):
    WS2812B = "ws2812b"
    SK6812 = "sk6812"
    APA102 = "apa102"
    WS2801 = "ws2801"
    VIRTUAL = "virtual"
    DUMMY = "dummy"

# LED strip interfaces
class LEDInterface(Enum):
    GPIO = "gpio"
    SPI = "spi"
    I2C = "i2c"
    PWM = "pwm"
    USB = "usb"
    VIRTUAL = "virtual"

# LED strip layouts
class LEDLayout(Enum):
    STRIP = "strip"
    MATRIX = "matrix"
    RING = "ring"
    CUSTOM = "custom"

# Color orders
class ColorOrder(Enum):
    RGB = "rgb"
    RBG = "rbg"
    GRB = "grb"
    GBR = "gbr"
    BRG = "brg"
    BGR = "bgr"

# Default colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

#--------------------------------------
#   LED CONTROLLER BASE CLASS
#--------------------------------------

class LEDController(ABC):
    """
    Abstract base class for LED controllers.
    
    This class defines the interface that all LED controllers must implement.
    """
    
    def __init__(self, config):
        """
        Initialize the LED controller.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.led_count = config.get("led_count", 60)
        self.brightness = config.get("brightness", 1.0)
        self.is_running = False
        self.pixels = [(0, 0, 0)] * self.led_count
        self.command_queue = queue.Queue(maxsize=100)
    
    @abstractmethod
    def setup(self):
        """
        Set up the LED hardware.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """
        Clean up the LED hardware.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def update(self):
        """
        Update the LED hardware with the current pixel data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    def start(self):
        """
        Start the LED controller.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.is_running:
            logger.warning("LED controller already running")
            return True
        
        success = self.setup()
        if success:
            self.is_running = True
            # Start the update thread
            self.update_thread = threading.Thread(target=self._update_thread, daemon=True)
            self.update_thread.start()
            
            logger.info("LED controller started")
            return True
        else:
            logger.error("Failed to start LED controller")
            return False
    
    def stop(self):
        """
        Stop the LED controller.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_running:
            logger.warning("LED controller already stopped")
            return True
        
        # Clear the LEDs
        self.set_all_pixels(BLACK)
        self.update()
        
        self.is_running = False
        # Wait for the update thread to finish
        if hasattr(self, 'update_thread') and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        
        success = self.cleanup()
        if success:
            logger.info("LED controller stopped")
            return True
        else:
            logger.error("Failed to stop LED controller")
            return False
    
    def set_pixel(self, index, color):
        """
        Set the color of a single pixel.
        
        Args:
            index (int): Pixel index
            color (tuple): RGB color value (0-255)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if 0 <= index < self.led_count:
            r, g, b = color
            # Clamp values to 0-255
            r = max(0, min(255, int(r * self.brightness)))
            g = max(0, min(255, int(g * self.brightness)))
            b = max(0, min(255, int(b * self.brightness)))
            
            self.pixels[index] = (r, g, b)
            return True
        else:
            return False
    
    def set_all_pixels(self, color):
        """
        Set all pixels to the same color.
        
        Args:
            color (tuple): RGB color value (0-255)
            
        Returns:
            bool: True if successful, False otherwise
        """
        r, g, b = color
        # Clamp values to 0-255
        r = max(0, min(255, int(r * self.brightness)))
        g = max(0, min(255, int(g * self.brightness)))
        b = max(0, min(255, int(b * self.brightness)))
        
        self.pixels = [(r, g, b)] * self.led_count
        return True
    
    def set_pixels(self, pixels):
        """
        Set all pixels to the specified colors.
        
        Args:
            pixels (list): List of RGB color values (0-255)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if len(pixels) != self.led_count:
            # Resize the input to match the LED count
            if len(pixels) > self.led_count:
                pixels = pixels[:self.led_count]
            else:
                pixels = pixels + [(0, 0, 0)] * (self.led_count - len(pixels))
        
        # Apply brightness and clamp values
        for i, (r, g, b) in enumerate(pixels):
            r = max(0, min(255, int(r * self.brightness)))
            g = max(0, min(255, int(g * self.brightness)))
            b = max(0, min(255, int(b * self.brightness)))
            
            self.pixels[i] = (r, g, b)
        
        return True
    
    def set_brightness(self, brightness):
        """
        Set the brightness of the LED strip.
        
        Args:
            brightness (float): Brightness value (0.0-1.0)
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.brightness = max(0.0, min(1.0, brightness))
        return True
    
    def queue_command(self, command, data=None):
        """
        Queue a command for the update thread.
        
        Args:
            command (str): Command name
            data: Command data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.command_queue.put((command, data), block=False)
            return True
        except queue.Full:
            logger.warning("Command queue is full")
            return False
    
    def _update_thread(self):
        """
        Thread function for updating the LEDs.
        """
        while self.is_running:
            try:
                # Process commands
                while not self.command_queue.empty():
                    try:
                        command, data = self.command_queue.get_nowait()
                        
                        if command == "set_pixel":
                            index, color = data
                            self.set_pixel(index, color)
                        
                        elif command == "set_all_pixels":
                            self.set_all_pixels(data)
                        
                        elif command == "set_pixels":
                            self.set_pixels(data)
                        
                        elif command == "set_brightness":
                            self.set_brightness(data)
                    
                    except queue.Empty:
                        break
                
                # Update the LEDs
                self.update()
                
                # Sleep for a short time to avoid maxing out CPU
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in LED update thread: {e}")
                time.sleep(0.1)

#--------------------------------------
#   NEOPIXEL LED CONTROLLER
#--------------------------------------

class NeoPixelController(LEDController):
    """
    Controller for NeoPixel (WS2812B) LED strips using the Adafruit NeoPixel library.
    """
    
    def __init__(self, config):
        """
        Initialize the NeoPixel controller.
        
        Args:
            config (dict): Configuration dictionary
        """
        super().__init__(config)
        self.pin = config.get("pin", board.D18)
        self.auto_write = config.get("auto_write", False)
        self.pixel_order = config.get("pixel_order", ColorOrder.GRB.value)
        self.strip = None
    
    def setup(self):
        """
        Set up the NeoPixel LED strip.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not HAS_NEOPIXEL:
            logger.error("NeoPixel library not available")
            return False
        
        try:
            # Create the NeoPixel object
            self.strip = neopixel.NeoPixel(
                self.pin,
                self.led_count,
                brightness=self.brightness,
                auto_write=self.auto_write,
                pixel_order=self.pixel_order
            )
            
            # Clear the strip
            self.set_all_pixels(BLACK)
            self.update()
            
            return True
        except Exception as e:
            logger.error(f"Error setting up NeoPixel LED strip: {e}")
            return False
    
    def cleanup(self):
        """
        Clean up the NeoPixel LED strip.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.strip is not None:
                # Clear the strip
                self.set_all_pixels(BLACK)
                self.update()
                
                # No explicit cleanup needed for NeoPixel
                self.strip = None
            
            return True
        except Exception as e:
            logger.error(f"Error cleaning up NeoPixel LED strip: {e}")
            return False
    
    def update(self):
        """
        Update the NeoPixel LED strip with the current pixel data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.strip is not None:
                for i, (r, g, b) in enumerate(self.pixels):
                    self.strip[i] = (r, g, b)
                
                # Only need to call show() if auto_write is False
                if not self.auto_write:
                    self.strip.show()
                
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error updating NeoPixel LED strip: {e}")
            return False

#--------------------------------------
#   DOTSTAR LED CONTROLLER
#--------------------------------------

class DotStarController(LEDController):
    """
    Controller for DotStar (APA102) LED strips using the Adafruit DotStar library.
    """
    
    def __init__(self, config):
        """
        Initialize the DotStar controller.
        
        Args:
            config (dict): Configuration dictionary
        """
        super().__init__(config)
        self.clock_pin = config.get("clock_pin", board.SCK)
        self.data_pin = config.get("data_pin", board.MOSI)
        self.auto_write = config.get("auto_write", False)
        self.pixel_order = config.get("pixel_order", ColorOrder.BGR.value)
        self.strip = None
    
    def setup(self):
        """
        Set up the DotStar LED strip.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not HAS_DOTSTAR:
            logger.error("DotStar library not available")
            return False
        
        try:
            # Create the DotStar object
            self.strip = adafruit_dotstar.DotStar(
                self.clock_pin,
                self.data_pin,
                self.led_count,
                brightness=self.brightness,
                auto_write=self.auto_write,
                pixel_order=self.pixel_order
            )
            
            # Clear the strip
            self.set_all_pixels(BLACK)
            self.update()
            
            return True
        except Exception as e:
            logger.error(f"Error setting up DotStar LED strip: {e}")
            return False
    
    def cleanup(self):
        """
        Clean up the DotStar LED strip.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.strip is not None:
                # Clear the strip
                self.set_all_pixels(BLACK)
                self.update()
                
                # Deinitialize
                self.strip.deinit()
                self.strip = None
            
            return True
        except Exception as e:
            logger.error(f"Error cleaning up DotStar LED strip: {e}")
            return False
    
    def update(self):
        """
        Update the DotStar LED strip with the current pixel data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.strip is not None:
                for i, (r, g, b) in enumerate(self.pixels):
                    self.strip[i] = (r, g, b)
                
                # Only need to call show() if auto_write is False
                if not self.auto_write:
                    self.strip.show()
                
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error updating DotStar LED strip: {e}")
            return False

#--------------------------------------
#   RPI_WS281X LED CONTROLLER
#--------------------------------------

class RPiWS281xController(LEDController):
    """
    Controller for WS281x LED strips using the rpi_ws281x library.
    
    This is a more low-level library that offers better performance on Raspberry Pi.
    """
    
    def __init__(self, config):
        """
        Initialize the RPiWS281x controller.
        
        Args:
            config (dict): Configuration dictionary
        """
        super().__init__(config)
        self.pin = config.get("pin", 18)  # PWM pin (hardware PWM on GPIO 18)
        self.freq_hz = config.get("freq_hz", 800000)  # 800kHz
        self.dma = config.get("dma", 10)
        self.invert = config.get("invert", False)
        self.channel = config.get("channel", 0)
        self.strip = None
    
    def setup(self):
        """
        Set up the WS281x LED strip.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not HAS_RPI_WS281X:
            logger.error("rpi_ws281x library not available")
            return False
        
        try:
            # Create the PixelStrip object
            self.strip = PixelStrip(
                self.led_count,
                self.pin,
                self.freq_hz,
                self.dma,
                self.invert,
                int(self.brightness * 255),
                self.channel
            )
            
            # Initialize the library
            self.strip.begin()
            
            # Clear the strip
            self.set_all_pixels(BLACK)
            self.update()
            
            return True
        except Exception as e:
            logger.error(f"Error setting up WS281x LED strip: {e}")
            return False
    
    def cleanup(self):
        """
        Clean up the WS281x LED strip.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.strip is not None:
                # Clear the strip
                self.set_all_pixels(BLACK)
                self.update()
                
                # No explicit cleanup needed for rpi_ws281x
                self.strip = None
            
            return True
        except Exception as e:
            logger.error(f"Error cleaning up WS281x LED strip: {e}")
            return False
    
    def update(self):
        """
        Update the WS281x LED strip with the current pixel data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.strip is not None:
                for i, (r, g, b) in enumerate(self.pixels):
                    self.strip.setPixelColor(i, Color(r, g, b))
                
                self.strip.show()
                
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error updating WS281x LED strip: {e}")
            return False
    
    def set_brightness(self, brightness):
        """
        Set the brightness of the LED strip.
        
        Args:
            brightness (float): Brightness value (0.0-1.0)
            
        Returns:
            bool: True if successful, False otherwise
        """
        brightness = max(0.0, min(1.0, brightness))
        
        try:
            if self.strip is not None:
                self.strip.setBrightness(int(brightness * 255))
                self.brightness = brightness
                return True
            else:
                self.brightness = brightness
                return False
        except Exception as e:
            logger.error(f"Error setting brightness: {e}")
            return False

#--------------------------------------
#   VIRTUAL LED CONTROLLER
#--------------------------------------

class VirtualLEDController(LEDController):
    """
    Virtual LED controller for testing and development.
    
    This controller visualizes the LEDs on the screen using Pygame.
    """
    
    def __init__(self, config):
        """
        Initialize the virtual LED controller.
        
        Args:
            config (dict): Configuration dictionary
        """
        super().__init__(config)
        self.width = config.get("width", 800)
        self.height = config.get("height", 600)
        self.led_size = config.get("led_size", 20)
        self.layout = config.get("layout", LEDLayout.STRIP.value)
        self.display = None
    
    def setup(self):
        """
        Set up the virtual LED display.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not HAS_PYGAME:
            logger.error("Pygame library not available")
            return False
        
        try:
            # Initialize Pygame
            pygame.init()
            
            # Set up the display
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Virtual LED Strip")
            
            # Clear the display
            self.display.fill((0, 0, 0))
            pygame.display.flip()
            
            return True
        except Exception as e:
            logger.error(f"Error setting up virtual LED display: {e}")
            return False
    
    def cleanup(self):
        """
        Clean up the virtual LED display.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.display is not None:
                # Clear the display
                self.display.fill((0, 0, 0))
                pygame.display.flip()
                
                # Quit Pygame
                pygame.quit()
                self.display = None
            
            return True
        except Exception as e:
            logger.error(f"Error cleaning up virtual LED display: {e}")
            return False
    
    def update(self):
        """
        Update the virtual LED display with the current pixel data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.display is not None:
                # Clear the display
                self.display.fill((0, 0, 0))
                
                # Calculate LED positions based on the layout
                led_positions = self._calculate_led_positions()
                
                # Draw the LEDs
                for i, (x, y) in enumerate(led_positions):
                    if i < len(self.pixels):
                        r, g, b = self.pixels[i]
                        # Draw an LED
                        pygame.draw.circle(self.display, (r, g, b), (x, y), self.led_size)
                
                # Update the display
                pygame.display.flip()
                
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.is_running = False
                
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error updating virtual LED display: {e}")
            return False
    
    def _calculate_led_positions(self):
        """
        Calculate the positions of the LEDs based on the layout.
        
        Returns:
            list: List of (x, y) positions for each LED
        """
        positions = []
        
        if self.layout == LEDLayout.STRIP.value:
            # Horizontal strip layout
            spacing = (self.width - 2 * self.led_size) / max(1, (self.led_count - 1))
            y = self.height // 2
            
            for i in range(self.led_count):
                x = self.led_size + i * spacing
                positions.append((int(x), y))
        
        elif self.layout == LEDLayout.MATRIX.value:
            # Matrix layout
            cols = int(np.sqrt(self.led_count))
            rows = (self.led_count + cols - 1) // cols
            
            spacing_x = (self.width - 2 * self.led_size) / max(1, (cols - 1))
            spacing_y = (self.height - 2 * self.led_size) / max(1, (rows - 1))
            
            for i in range(self.led_count):
                row = i // cols
                col = i % cols
                
                # Zigzag pattern
                if row % 2 == 1:
                    col = cols - 1 - col
                
                x = self.led_size + col * spacing_x
                y = self.led_size + row * spacing_y
                
                positions.append((int(x), int(y)))
        
        elif self.layout == LEDLayout.RING.value:
            # Ring layout
            center_x = self.width // 2
            center_y = self.height // 2
            radius = min(self.width, self.height) // 2 - self.led_size
            
            for i in range(self.led_count):
                angle = 2 * np.pi * i / self.led_count
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                
                positions.append((int(x), int(y)))
        
        else:  # LEDLayout.CUSTOM or unknown
            # Default to a horizontal strip
            spacing = (self.width - 2 * self.led_size) / max(1, (self.led_count - 1))
            y = self.height // 2
            
            for i in range(self.led_count):
                x = self.led_size + i * spacing
                positions.append((int(x), y))
        
        return positions

#--------------------------------------
#   DUMMY LED CONTROLLER
#--------------------------------------

class DummyLEDController(LEDController):
    """
    Dummy LED controller that doesn't control any hardware.
    
    Useful for testing or when no LED hardware is available.
    """
    
    def setup(self):
        """
        Set up the dummy LED controller.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Nothing to set up
        logger.info("Dummy LED controller initialized")
        return True
    
    def cleanup(self):
        """
        Clean up the dummy LED controller.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Nothing to clean up
        return True
    
    def update(self):
        """
        Update the dummy LED controller with the current pixel data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Nothing to update
        time.sleep(0.01)  # Small delay to avoid maxing out CPU
        return True

#--------------------------------------
#   LED MANAGER
#--------------------------------------

class LEDManager:
    """
    LED manager for the Audio LED Visualization System.
    
    This class manages the LED controller and provides
    high-level LED control functionality.
    """
    
    def __init__(self, config):
        """
        Initialize the LED manager.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Get LED configuration
        self.led_config = config.get("led", {})
        self.strip_type = LEDStripType(self.led_config.get("strip_type", "virtual"))
        
        # Initialize variables
        self.controller = None
        self.is_running = False
        
        # LED parameters
        self.led_count = self.led_config.get("led_count", 60)
        self.brightness = self.led_config.get("brightness", 1.0)
        self.gamma_correction = self.led_config.get("gamma_correction", 2.2)
        
        # Animation parameters
        self.animation_fps = self.led_config.get("animation_fps", 30)
        self.animation_speed = self.led_config.get("animation_speed", 1.0)
        
        # Create gamma correction table
        self.gamma_table = [int(255 * (i / 255) ** self.gamma_correction) for i in range(256)]
    
    def start(self):
        """
        Start the LED manager.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.is_running:
            logger.warning("LED manager already running")
            return True
        
        # Initialize the LED controller
        if self.controller is not None:
            self.controller.stop()
        
        try:
            # Create the appropriate controller based on strip type
            if self.strip_type == LEDStripType.WS2812B:
                if HAS_NEOPIXEL:
                    self.controller = NeoPixelController(self.led_config)
                elif HAS_RPI_WS281X:
                    self.controller = RPiWS281xController(self.led_config)
                else:
                    logger.warning("No WS2812B library available, using dummy controller")
                    self.controller = DummyLEDController(self.led_config)
            
            elif self.strip_type == LEDStripType.APA102:
                if HAS_DOTSTAR:
                    self.controller = DotStarController(self.led_config)
                else:
                    logger.warning("DotStar library not available, using dummy controller")
                    self.controller = DummyLEDController(self.led_config)
            
            elif self.strip_type == LEDStripType.VIRTUAL:
                self.controller = VirtualLEDController(self.led_config)
            
            elif self.strip_type == LEDStripType.DUMMY:
                self.controller = DummyLEDController(self.led_config)
            
            else:
                logger.warning(f"Unsupported LED strip type: {self.strip_type}, using dummy controller")
                self.controller = DummyLEDController(self.led_config)
        
        except Exception as e:
            logger.error(f"Error initializing LED controller: {e}")
            logger.warning("Using dummy controller as fallback")
            self.controller = DummyLEDController(self.led_config)
        
        # Start the controller
        if not self.controller.start():
            logger.error("Failed to start LED controller")
            return False
        
        self.is_running = True
        logger.info(f"LED manager started with {self.strip_type.value} controller")
        
        # Display test pattern
        self._display_test_pattern()
        
        return True
    
    def stop(self):
        """
        Stop the LED manager.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_running:
            logger.warning("LED manager already stopped")
            return True
        
        self.is_running = False
        
        # Stop the controller
        if self.controller is not None:
            self.controller.stop()
        
        logger.info("LED manager stopped")
        return True
    
    def set_pixels(self, pixels):
        """
        Set the LED pixels.
        
        Args:
            pixels (list): List of RGB color values (0-255)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_running or self.controller is None:
            return False
        
        # Apply gamma correction
        gamma_corrected_pixels = []
        for r, g, b in pixels:
            gamma_r = self.gamma_table[min(255, max(0, int(r)))]
            gamma_g = self.gamma_table[min(255, max(0, int(g)))]
            gamma_b = self.gamma_table[min(255, max(0, int(b)))]
            gamma_corrected_pixels.append((gamma_r, gamma_g, gamma_b))
        
        return self.controller.queue_command("set_pixels", gamma_corrected_pixels)
    
    def set_all_pixels(self, color):
        """
        Set all pixels to the same color.
        
        Args:
            color (tuple): RGB color value (0-255)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_running or self.controller is None:
            return False
        
        # Apply gamma correction
        r, g, b = color
        gamma_r = self.gamma_table[min(255, max(0, int(r)))]
        gamma_g = self.gamma_table[min(255, max(0, int(g)))]
        gamma_b = self.gamma_table[min(255, max(0, int(b)))]
        gamma_corrected_color = (gamma_r, gamma_g, gamma_b)
        
        return self.controller.queue_command("set_all_pixels", gamma_corrected_color)
    
    def set_brightness(self, brightness):
        """
        Set the brightness of the LED strip.
        
        Args:
            brightness (float): Brightness value (0.0-1.0)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_running or self.controller is None:
            return False
        
        self.brightness = max(0.0, min(1.0, brightness))
        return self.controller.queue_command("set_brightness", self.brightness)
    
    def _display_test_pattern(self):
        """
        Display a test pattern on the LED strip.
        
        This helps verify that the LED strip is working correctly.
        """
        if not self.is_running or self.controller is None:
            return
        
        # Display red, green, blue, and white in sequence
        for color in [RED, GREEN, BLUE, WHITE]:
            self.set_all_pixels(color)
            time.sleep(0.5)
        
        # Rainbow pattern
        hue_step = 360 / self.led_count
        pixels = []
        for i in range(self.led_count):
            hue = (i * hue_step) % 360
            r, g, b = self._hsv_to_rgb(hue, 1.0, 1.0)
            pixels.append((r, g, b))
        
        self.set_pixels(pixels)
    
    @staticmethod
    def _hsv_to_rgb(h, s, v):
        """
        Convert HSV color to RGB.
        
        Args:
            h (float): Hue (0-360)
            s (float): Saturation (0-1)
            v (float): Value (0-1)
            
        Returns:
            tuple: RGB color (0-255)
        """
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)
        
        return r, g, b

# Test the module if run directly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    test_config = {
        "led": {
            "strip_type": "virtual",
            "led_count": 60,
            "brightness": 0.5,
            "layout": "ring",
            "width": 800,
            "height": 600,
            "led_size": 10
        }
    }
    
    # Create the LED manager
    manager = LEDManager(test_config)
    
    # Start the manager
    manager.start()
    
    try:
        # Animation loop
        time_start = time.time()
        
        while True:
            # Calculate time offset
            t = time.time() - time_start
            
            # Create a rainbow pattern that moves over time
            pixels = []
            for i in range(manager.led_count):
                # Rainbow pattern with movement
                hue = (i * 360 / manager.led_count + t * 50) % 360
                r, g, b = manager._hsv_to_rgb(hue, 1.0, 1.0)
                pixels.append((r, g, b))
            
            # Update the LEDs
            manager.set_pixels(pixels)
            
            # Sleep to maintain frame rate
            time.sleep(1.0 / 30)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        # Stop the manager
        manager.stop() 