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

# This module handles visual output for the Audio LED Visualization System.
# It provides interfaces to different output methods:
# - LED control via GPIO pins (direct PWM for RGB LEDs)
# - LED strip control (WS2812, NeoPixels, etc.)
# - Display output for visualization on screen
# - File output for debugging or testing
#
# The module adapts to different hardware platforms (Raspberry Pi, Pico W, Unix)
# and provides a unified interface for all output methods.

#--------------------------------------
#       IMPORTS
#--------------------------------------

import os
import sys
import logging
import time
import threading
from pathlib import Path
from abc import ABC, abstractmethod

# Import display modules if available
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    logging.debug("pygame not available, display output will be limited")

# Configure module logger
logger = logging.getLogger(__name__)

#--------------------------------------
#       CONSTANTS
#--------------------------------------

# Output types
OUTPUT_LED_PWM = "led_pwm"        # Direct PWM control of RGB LEDs
OUTPUT_LED_STRIP = "led_strip"    # WS2812/NeoPixel LED strips
OUTPUT_DISPLAY = "display"        # On-screen visualization
OUTPUT_FILE = "file"              # File output for debugging/testing
OUTPUT_NONE = "none"              # No output (for testing)

# Default LED pins for different platforms
DEFAULT_PINS = {
    "raspberry_pi": {
        "red_pin": 17,
        "green_pin": 22,
        "blue_pin": 24
    },
    "pico_w": {
        "red_pin": 17,
        "green_pin": 22,
        "blue_pin": 16
    },
    "unix": {
        "red_pin": 0,
        "green_pin": 0,
        "blue_pin": 0
    }
}

# Default display settings
DEFAULT_DISPLAY_WIDTH = 800
DEFAULT_DISPLAY_HEIGHT = 600
DEFAULT_DISPLAY_FPS = 30

#--------------------------------------
#       BASE CLASSES
#--------------------------------------

class OutputDevice(ABC):
    """
    Abstract base class for output devices.
    
    This class defines the interface for all output devices.
    """
    
    def __init__(self, env_config):
        """
        Initialize the output device.
        
        Args:
            env_config (dict): Environment configuration
        """
        self.env_config = env_config
        self.is_open = False
    
    @abstractmethod
    def open(self):
        """
        Open the output device.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def update(self, rgb):
        """
        Update the output device with new RGB values.
        
        Args:
            rgb (tuple): RGB color tuple (R, G, B)
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def close(self):
        """
        Close the output device and clean up resources.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass

#--------------------------------------
#       OUTPUT DEVICE IMPLEMENTATIONS
#--------------------------------------

class PWMLEDOutput(OutputDevice):
    """
    Output device for controlling RGB LEDs via PWM.
    
    This class controls individual RGB LEDs using PWM signals
    on GPIO pins.
    """
    
    def __init__(self, env_config):
        """
        Initialize the PWM LED output device.
        
        Args:
            env_config (dict): Environment configuration
        """
        super().__init__(env_config)
        
        # Get hardware configuration
        self.hardware_config = env_config.get('config', {}).get('hardware', {})
        self.system_type = env_config.get('env_info', {}).get('system_type', 'unix')
        
        # Get pin assignments
        default_pins = DEFAULT_PINS.get(self.system_type, DEFAULT_PINS['unix'])
        self.red_pin = self.hardware_config.get('red_pin', default_pins['red_pin'])
        self.green_pin = self.hardware_config.get('green_pin', default_pins['green_pin'])
        self.blue_pin = self.hardware_config.get('blue_pin', default_pins['blue_pin'])
        
        # Set up GPIO interface based on system type
        self.gpio = None
        self.pi = None  # For pigpio
        
        logger.info(f"PWM LED output initialized (red={self.red_pin}, green={self.green_pin}, blue={self.blue_pin})")
    
    def open(self):
        """
        Open the PWM LED output device.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.is_open:
            return True
        
        try:
            # Initialize GPIO based on system type
            if self.system_type == 'raspberry_pi':
                # Use pigpio for hardware PWM (better quality)
                import pigpio
                self.pi = pigpio.pi()
                if not self.pi.connected:
                    logger.error("Could not connect to pigpio daemon")
                    return False
                
                # Set up pins
                for pin in [self.red_pin, self.green_pin, self.blue_pin]:
                    self.pi.set_mode(pin, pigpio.OUTPUT)
                    self.pi.set_PWM_frequency(pin, 800)  # 800 Hz frequency
                
                self.gpio = 'pigpio'
                
            elif self.system_type == 'pico_w':
                # Import MicroPython libraries
                from machine import Pin, PWM
                
                # Set up PWM pins
                self.red_led = PWM(Pin(self.red_pin))
                self.green_led = PWM(Pin(self.green_pin))
                self.blue_led = PWM(Pin(self.blue_pin))
                
                # Set frequency to 1000 Hz
                self.red_led.freq(1000)
                self.green_led.freq(1000)
                self.blue_led.freq(1000)
                
                self.gpio = 'micropython'
                
            else:
                # Fallback - don't actually control LEDs
                logger.warning("GPIO control not available on this system")
                self.gpio = 'stub'
            
            self.is_open = True
            logger.info(f"PWM LED output opened using {self.gpio}")
            return True
            
        except ImportError as e:
            logger.error(f"Error importing GPIO library: {e}")
            return False
        except Exception as e:
            logger.error(f"Error opening PWM LED output: {e}")
            return False
    
    def update(self, rgb):
        """
        Update the PWM LED output with new RGB values.
        
        Args:
            rgb (tuple): RGB color tuple (R, G, B)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_open:
            if not self.open():
                return False
        
        try:
            r, g, b = rgb
            
            if self.gpio == 'pigpio':
                # Set PWM duty cycle (0-255)
                self.pi.set_PWM_dutycycle(self.red_pin, r)
                self.pi.set_PWM_dutycycle(self.green_pin, g)
                self.pi.set_PWM_dutycycle(self.blue_pin, b)
                
            elif self.gpio == 'micropython':
                # Convert 0-255 to 0-65535 (16-bit PWM on RP2040)
                r_duty = int(r * 257)  # 257 = 65535/255
                g_duty = int(g * 257)
                b_duty = int(b * 257)
                
                self.red_led.duty_u16(r_duty)
                self.green_led.duty_u16(g_duty)
                self.blue_led.duty_u16(b_duty)
                
            elif self.gpio == 'stub':
                # Just log the values for debugging
                if hasattr(self, 'last_log_time') and time.time() - self.last_log_time < 1.0:
                    # Don't log too frequently
                    pass
                else:
                    logger.debug(f"LED would be set to RGB: ({r}, {g}, {b})")
                    self.last_log_time = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating PWM LED output: {e}")
            return False
    
    def close(self):
        """
        Close the PWM LED output and clean up resources.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_open:
            return True
        
        try:
            # Turn off all LEDs
            self.update((0, 0, 0))
            
            # Clean up GPIO
            if self.gpio == 'pigpio' and self.pi:
                self.pi.stop()
                
            elif self.gpio == 'micropython':
                # Deinitialize PWM
                self.red_led.deinit()
                self.green_led.deinit()
                self.blue_led.deinit()
            
            self.is_open = False
            logger.info("PWM LED output closed")
            return True
            
        except Exception as e:
            logger.error(f"Error closing PWM LED output: {e}")
            return False


class LEDStripOutput(OutputDevice):
    """
    Output device for controlling LED strips.
    
    This class controls addressable LED strips like WS2812/NeoPixels.
    """
    
    def __init__(self, env_config):
        """
        Initialize the LED strip output device.
        
        Args:
            env_config (dict): Environment configuration
        """
        super().__init__(env_config)
        
        # Get hardware configuration
        self.hardware_config = env_config.get('config', {}).get('hardware', {})
        self.system_type = env_config.get('env_info', {}).get('system_type', 'unix')
        
        # Get LED strip configuration
        self.led_type = self.hardware_config.get('led_type', 'ws2812')
        self.led_count = self.hardware_config.get('led_count', 60)
        self.led_pin = self.hardware_config.get('led_pin', 18)  # Default pin for LED strips
        
        # Initialize LED strip library
        self.strip = None
        
        logger.info(f"LED strip output initialized (type={self.led_type}, count={self.led_count}, pin={self.led_pin})")
    
    def open(self):
        """
        Open the LED strip output device.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.is_open:
            return True
        
        try:
            # Initialize LED strip based on system type and LED type
            if self.system_type == 'raspberry_pi':
                if self.led_type == 'ws2812':
                    # Try to import the rpi_ws281x library
                    try:
                        from rpi_ws281x import PixelStrip, Color
                        
                        # Create NeoPixel object
                        self.strip = PixelStrip(
                            num=self.led_count,
                            pin=self.led_pin,
                            freq_hz=800000,
                            dma=10,
                            invert=False,
                            brightness=255,
                            channel=0
                        )
                        
                        # Initialize the library
                        self.strip.begin()
                        
                    except ImportError:
                        logger.error("rpi_ws281x library not found")
                        return False
                
                elif self.led_type == 'apa102':
                    # Try to import the APA102 library
                    try:
                        import apa102
                        
                        # Create APA102 object
                        self.strip = apa102.APA102(
                            num_led=self.led_count,
                            global_brightness=31,  # Maximum
                            mosi=10,  # MOSI pin
                            sclk=11,  # SCLK pin
                            order='rgb'
                        )
                        
                    except ImportError:
                        logger.error("APA102 library not found")
                        return False
            
            elif self.system_type == 'pico_w':
                # Try to import the MicroPython NeoPixel library
                try:
                    from machine import Pin
                    from neopixel import NeoPixel
                    
                    # Create NeoPixel object
                    self.strip = NeoPixel(Pin(self.led_pin), self.led_count)
                    
                except ImportError:
                    logger.error("NeoPixel library not found for MicroPython")
                    return False
            
            else:
                # Fallback - don't actually control LEDs
                logger.warning("LED strip control not available on this system")
                self.strip = 'stub'
            
            self.is_open = True
            logger.info(f"LED strip output opened")
            return True
            
        except Exception as e:
            logger.error(f"Error opening LED strip output: {e}")
            return False
    
    def update(self, rgb):
        """
        Update the LED strip output with new RGB values.
        
        Args:
            rgb (tuple): RGB color tuple (R, G, B)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_open:
            if not self.open():
                return False
        
        try:
            r, g, b = rgb
            
            if self.strip == 'stub':
                # Just log the values for debugging
                if hasattr(self, 'last_log_time') and time.time() - self.last_log_time < 1.0:
                    # Don't log too frequently
                    pass
                else:
                    logger.debug(f"LED strip would be set to RGB: ({r}, {g}, {b})")
                    self.last_log_time = time.time()
                return True
            
            # Set all LEDs to the same color for now
            # (could be extended for more complex patterns)
            if isinstance(self.strip, str):
                return True
                
            # Handle different types of LED strips
            if self.system_type == 'raspberry_pi':
                if self.led_type == 'ws2812':
                    from rpi_ws281x import Color
                    color = Color(r, g, b)
                    
                    for i in range(self.led_count):
                        self.strip.setPixelColor(i, color)
                    
                    self.strip.show()
                    
                elif self.led_type == 'apa102':
                    for i in range(self.led_count):
                        self.strip.set_pixel(i, r, g, b)
                    
                    self.strip.show()
            
            elif self.system_type == 'pico_w':
                for i in range(self.led_count):
                    self.strip[i] = (r, g, b)
                
                self.strip.write()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating LED strip output: {e}")
            return False
    
    def close(self):
        """
        Close the LED strip output and clean up resources.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_open:
            return True
        
        try:
            # Turn off all LEDs
            self.update((0, 0, 0))
            
            # Clean up resources
            if self.strip != 'stub':
                # No special cleanup for most LED strip libraries
                pass
            
            self.is_open = False
            logger.info("LED strip output closed")
            return True
            
        except Exception as e:
            logger.error(f"Error closing LED strip output: {e}")
            return False


class DisplayOutput(OutputDevice):
    """
    Output device for on-screen visualization.
    
    This class displays the RGB colors on the screen using pygame
    or other display libraries.
    """
    
    def __init__(self, env_config):
        """
        Initialize the display output device.
        
        Args:
            env_config (dict): Environment configuration
        """
        super().__init__(env_config)
        
        # Get display configuration
        self.display_config = env_config.get('config', {}).get('visual', {})
        
        # Initialize display settings
        self.width = self.display_config.get('display_width', DEFAULT_DISPLAY_WIDTH)
        self.height = self.display_config.get('display_height', DEFAULT_DISPLAY_HEIGHT)
        self.fps = self.display_config.get('refresh_rate', DEFAULT_DISPLAY_FPS)
        
        # Initialize pygame variables
        self.screen = None
        self.clock = None
        self.running = False
        self.display_thread = None
        
        # Queue for communication between threads
        self.current_rgb = (0, 0, 0)
        self.update_lock = threading.Lock()
        
        logger.info(f"Display output initialized (width={self.width}, height={self.height}, fps={self.fps})")
    
    def _display_thread_func(self):
        """
        Thread function for updating the display.
        """
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Audio LED Visualization")
        
        # Create the screen
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Font for text
        try:
            self.font = pygame.font.Font(None, 36)
        except:
            self.font = pygame.font.SysFont('Arial', 36)
        
        # Main loop
        self.running = True
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            # Get current RGB values
            with self.update_lock:
                rgb = self.current_rgb
            
            # Clear the screen
            self.screen.fill((0, 0, 0))
            
            # Draw the color rectangle
            pygame.draw.rect(self.screen, rgb, pygame.Rect(0, 0, self.width, self.height))
            
            # Draw RGB values as text
            text = f"RGB: ({rgb[0]}, {rgb[1]}, {rgb[2]})"
            text_surface = self.font.render(text, True, (255, 255, 255) if sum(rgb) < 380 else (0, 0, 0))
            text_rect = text_surface.get_rect(center=(self.width//2, self.height//2))
            self.screen.blit(text_surface, text_rect)
            
            # Update the display
            pygame.display.flip()
            
            # Cap the frame rate
            self.clock.tick(self.fps)
        
        # Clean up pygame
        pygame.quit()
    
    def open(self):
        """
        Open the display output device.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.is_open:
            return True
        
        try:
            # Check if pygame is available
            if not HAS_PYGAME:
                logger.error("pygame not available, cannot use display output")
                return False
            
            # Start the display thread
            self.display_thread = threading.Thread(target=self._display_thread_func)
            self.display_thread.daemon = True
            self.display_thread.start()
            
            self.is_open = True
            logger.info("Display output opened")
            return True
            
        except Exception as e:
            logger.error(f"Error opening display output: {e}")
            return False
    
    def update(self, rgb):
        """
        Update the display with new RGB values.
        
        Args:
            rgb (tuple): RGB color tuple (R, G, B)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_open:
            if not self.open():
                return False
        
        try:
            # Update the RGB values
            with self.update_lock:
                self.current_rgb = rgb
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating display output: {e}")
            return False
    
    def close(self):
        """
        Close the display output and clean up resources.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_open:
            return True
        
        try:
            # Stop the display thread
            self.running = False
            if self.display_thread and self.display_thread.is_alive():
                self.display_thread.join(1.0)  # Wait for up to 1 second
            
            self.is_open = False
            logger.info("Display output closed")
            return True
            
        except Exception as e:
            logger.error(f"Error closing display output: {e}")
            return False


class FileOutput(OutputDevice):
    """
    Output device for writing RGB values to a file.
    
    This class writes RGB values to a file for debugging or testing.
    """
    
    def __init__(self, env_config):
        """
        Initialize the file output device.
        
        Args:
            env_config (dict): Environment configuration
        """
        super().__init__(env_config)
        
        # Get file configuration
        self.file_config = env_config.get('config', {}).get('visual', {})
        
        # Initialize file settings
        self.output_file = self.file_config.get('output_file', None)
        
        # Set default output file if none specified
        if self.output_file is None:
            self.output_file = os.path.join(os.getcwd(), 'rgb_output.txt')
            
        self.file_handle = None
        
        logger.info(f"File output initialized (file={self.output_file})")
    
    def open(self):
        """
        Open the file output device.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.is_open:
            return True
        
        try:
            # Open the output file
            self.file_handle = open(self.output_file, 'w')
            self.file_handle.write("Red\tGreen\tBlue\tTimestamp\n")
            
            self.is_open = True
            logger.info(f"File output opened (file={self.output_file})")
            return True
            
        except Exception as e:
            logger.error(f"Error opening file output: {e}")
            return False
    
    def update(self, rgb):
        """
        Update the file with new RGB values.
        
        Args:
            rgb (tuple): RGB color tuple (R, G, B)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_open:
            if not self.open():
                return False
        
        try:
            # Write RGB values to the file
            timestamp = time.time()
            self.file_handle.write(f"{rgb[0]}\t{rgb[1]}\t{rgb[2]}\t{timestamp}\n")
            self.file_handle.flush()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating file output: {e}")
            return False
    
    def close(self):
        """
        Close the file output and clean up resources.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_open:
            return True
        
        try:
            # Close the output file
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None
            
            self.is_open = False
            logger.info("File output closed")
            return True
            
        except Exception as e:
            logger.error(f"Error closing file output: {e}")
            return False

#--------------------------------------
#       MAIN OUTPUT HANDLER
#--------------------------------------

class OutputHandler:
    """
    Main handler for managing multiple output devices.
    
    This class provides a unified interface for controlling multiple
    output devices, automatically selecting the appropriate devices
    based on the environment configuration.
    """
    
    def __init__(self, env_config, device_manager=None):
        """
        Initialize the output handler.
        
        Args:
            env_config (dict): Environment configuration
            device_manager (DeviceManager, optional): Device manager for hardware access
        """
        self.env_config = env_config
        self.device_manager = device_manager
        
        # Get output configuration
        self.visual_config = env_config.get('config', {}).get('visual', {})
        
        # Initialize output devices
        self.outputs = []
        self.output_method = self.visual_config.get('output_method', 'auto')
        
        # Auto-detect output method if needed
        if self.output_method == 'auto':
            self._auto_detect_output_method()
        
        # Initialize output devices based on the selected method
        self._initialize_outputs()
        
        logger.info(f"Output handler initialized (method={self.output_method}, outputs={len(self.outputs)})")
    
    def _auto_detect_output_method(self):
        """
        Auto-detect the appropriate output method based on the environment.
        """
        system_type = self.env_config.get('env_info', {}).get('system_type', 'unix')
        capabilities = self.env_config.get('env_info', {}).get('capabilities', {})
        
        # Check for hardware capabilities
        if system_type in ['raspberry_pi', 'pico_w'] and capabilities.get('gpio', False):
            self.output_method = OUTPUT_LED_PWM
        elif capabilities.get('display', False):
            self.output_method = OUTPUT_DISPLAY
        else:
            self.output_method = OUTPUT_FILE
        
        logger.info(f"Auto-detected output method: {self.output_method}")
    
    def _initialize_outputs(self):
        """
        Initialize output devices based on the selected method.
        """
        # Create the appropriate output devices
        if self.output_method == OUTPUT_LED_PWM:
            self.outputs.append(PWMLEDOutput(self.env_config))
            
        elif self.output_method == OUTPUT_LED_STRIP:
            self.outputs.append(LEDStripOutput(self.env_config))
            
        elif self.output_method == OUTPUT_DISPLAY:
            if HAS_PYGAME:
                self.outputs.append(DisplayOutput(self.env_config))
            else:
                logger.warning("pygame not available, falling back to file output")
                self.outputs.append(FileOutput(self.env_config))
                
        elif self.output_method == OUTPUT_FILE:
            self.outputs.append(FileOutput(self.env_config))
            
        elif self.output_method == OUTPUT_NONE:
            logger.info("No output method selected")
            
        else:
            logger.warning(f"Unknown output method: {self.output_method}")
    
    def update(self, rgb):
        """
        Update all output devices with new RGB values.
        
        Args:
            rgb (tuple): RGB color tuple (R, G, B)
            
        Returns:
            bool: True if all outputs were updated successfully, False otherwise
        """
        success = True
        
        for output in self.outputs:
            if not output.update(rgb):
                success = False
        
        return success
    
    def close(self):
        """
        Close all output devices and clean up resources.
        
        Returns:
            bool: True if all outputs were closed successfully, False otherwise
        """
        success = True
        
        for output in self.outputs:
            if not output.close():
                success = False
        
        return success

# Test the module if run directly
if __name__ == "__main__":
    import time
    import math
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple environment configuration for testing
    env_config = {
        'config': {
            'visual': {
                'output_method': 'auto',
                'output_file': 'test_rgb_output.txt',
                'display_width': 400,
                'display_height': 300,
                'refresh_rate': 30
            },
            'hardware': {
                'led_type': 'ws2812',
                'led_count': 60,
                'led_pin': 18,
                'red_pin': 17,
                'green_pin': 22,
                'blue_pin': 24
            }
        },
        'env_info': {
            'system_type': 'unix',
            'capabilities': {
                'gpio': False,
                'display': True
            }
        }
    }
    
    # Initialize the output handler
    output_handler = OutputHandler(env_config)
    
    try:
        # Generate a rainbow of colors
        print("Generating a rainbow of colors for 10 seconds...")
        
        start_time = time.time()
        while time.time() - start_time < 10:
            # Calculate rainbow color based on time
            hue = (time.time() * 0.1) % 1.0
            
            # Convert HSV to RGB
            if hue < 1/6:
                r = 1
                g = hue * 6
                b = 0
            elif hue < 2/6:
                r = (2/6 - hue) * 6
                g = 1
                b = 0
            elif hue < 3/6:
                r = 0
                g = 1
                b = (hue - 2/6) * 6
            elif hue < 4/6:
                r = 0
                g = (4/6 - hue) * 6
                b = 1
            elif hue < 5/6:
                r = (hue - 4/6) * 6
                b = 1
                g = 0
            else:
                r = 1
                g = 0
                b = (1 - hue) * 6
            
            # Scale to 0-255
            rgb = (int(r * 255), int(g * 255), int(b * 255))
            
            # Update the output
            output_handler.update(rgb)
            
            # Print the color
            if hasattr(output_handler, 'last_print_time') and time.time() - output_handler.last_print_time < 0.5:
                # Don't print too frequently
                pass
            else:
                print(f"RGB: {rgb}")
                output_handler.last_print_time = time.time()
            
            # Small delay
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Turn off LEDs and close the output
        output_handler.update((0, 0, 0))
        output_handler.close() 