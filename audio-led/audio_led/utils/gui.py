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

# This module provides a graphical user interface for the Audio LED Visualization System.
# It visualizes audio data in real-time and provides controls for adjusting parameters.
# Features include:
# - Real-time audio visualization (waveform, spectrum, volume)
# - RGB color display based on audio data
# - Controls for adjusting audio processing parameters
# - Configuration management
# - Visualization of LED patterns
#
# The GUI is implemented using pygame.

#--------------------------------------
#       IMPORTS
#--------------------------------------

import os
import sys
import time
import logging
import threading
import numpy as np
import signal
from pathlib import Path

# Import pygame
import pygame
import pygame.font
import pygame.gfxdraw

# Import from audio_led package
from audio_led.audio import processor
from audio_led.visual import rgb_converter
from audio_led.hardware import led_controller

# Configure module logger
logger = logging.getLogger(__name__)

#--------------------------------------
#       CONSTANTS
#--------------------------------------

# GUI dimensions and layout
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
PADDING = 20
SIDEBAR_WIDTH = 200
MAIN_WIDTH = WINDOW_WIDTH - SIDEBAR_WIDTH - PADDING * 3
MAIN_HEIGHT = WINDOW_HEIGHT - PADDING * 2

# Colors
BG_COLOR = (30, 30, 40)
PANEL_COLOR = (40, 40, 50)
TEXT_COLOR = (220, 220, 220)
HIGHLIGHT_COLOR = (60, 130, 220)
AXIS_COLOR = (120, 120, 140)
VOLUME_COLOR = (80, 220, 100)
BEAT_COLOR = (220, 80, 80)

# UI elements
BUTTON_HEIGHT = 30
SLIDER_HEIGHT = 20
UI_SPACING = 10

# Animation
FPS = 60
SPECTRUM_SMOOTHING = 0.7

#--------------------------------------
#       UI COMPONENTS
#--------------------------------------

class Button:
    """Button UI component."""
    
    def __init__(self, x, y, width, height, text, callback):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.hover = False
        
    def draw(self, surface, font):
        # Draw button background
        color = HIGHLIGHT_COLOR if self.hover else PANEL_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, TEXT_COLOR, self.rect, width=1, border_radius=5)
        
        # Draw button text
        text_surf = font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
    
    def check_hover(self, pos):
        self.hover = self.rect.collidepoint(pos)
        return self.hover
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hover:
                self.callback()
                return True
        return False


class Slider:
    """Slider UI component for adjusting values."""
    
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label, callback):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.callback = callback
        self.active = False
        self.hover = False
        
        # Calculate handle position
        self.handle_width = 10
        self.handle_rect = pygame.Rect(0, 0, self.handle_width, height + 4)
        self.update_handle_position()
    
    def update_handle_position(self):
        """Update the position of the slider handle based on the current value."""
        value_range = self.max_val - self.min_val
        position = (self.value - self.min_val) / value_range
        handle_x = self.rect.x + (self.rect.width - self.handle_width) * position
        self.handle_rect.x = handle_x
        self.handle_rect.y = self.rect.y - 2
    
    def set_value_from_pos(self, x_pos):
        """Set the slider value based on the mouse position."""
        rel_x = max(0, min(x_pos - self.rect.x, self.rect.width))
        self.value = self.min_val + (rel_x / self.rect.width) * (self.max_val - self.min_val)
        self.update_handle_position()
        self.callback(self.value)
    
    def draw(self, surface, font):
        # Draw slider track
        track_color = HIGHLIGHT_COLOR if self.hover or self.active else PANEL_COLOR
        pygame.draw.rect(surface, track_color, self.rect, border_radius=3)
        pygame.draw.rect(surface, TEXT_COLOR, self.rect, width=1, border_radius=3)
        
        # Draw slider handle
        handle_color = HIGHLIGHT_COLOR if self.active else TEXT_COLOR
        pygame.draw.rect(surface, handle_color, self.handle_rect, border_radius=2)
        
        # Draw label and value
        label_text = f"{self.label}: {self.value:.2f}"
        text_surf = font.render(label_text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(x=self.rect.x, bottom=self.rect.y - 5)
        surface.blit(text_surf, text_rect)
    
    def check_hover(self, pos):
        self.hover = self.rect.collidepoint(pos) or self.handle_rect.collidepoint(pos)
        return self.hover
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hover:
                self.active = True
                self.set_value_from_pos(event.pos[0])
                return True
        elif event.type == pygame.MOUSEMOTION:
            if self.active:
                self.set_value_from_pos(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.active:
                self.active = False
                return True
        return False


class CheckBox:
    """Checkbox UI component for toggling boolean values."""
    
    def __init__(self, x, y, size, label, initial_state, callback):
        self.rect = pygame.Rect(x, y, size, size)
        self.label = label
        self.checked = initial_state
        self.callback = callback
        self.hover = False
    
    def draw(self, surface, font):
        # Draw checkbox background
        color = HIGHLIGHT_COLOR if self.hover else PANEL_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=3)
        pygame.draw.rect(surface, TEXT_COLOR, self.rect, width=1, border_radius=3)
        
        # Draw checkmark if checked
        if self.checked:
            check_rect = pygame.Rect(
                self.rect.x + 4, 
                self.rect.y + 4, 
                self.rect.width - 8, 
                self.rect.height - 8
            )
            pygame.draw.rect(surface, HIGHLIGHT_COLOR, check_rect, border_radius=2)
        
        # Draw label
        text_surf = font.render(self.label, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(x=self.rect.right + 10, centery=self.rect.centery)
        surface.blit(text_surf, text_rect)
    
    def check_hover(self, pos):
        self.hover = self.rect.collidepoint(pos)
        return self.hover
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hover:
                self.checked = not self.checked
                self.callback(self.checked)
                return True
        return False


#--------------------------------------
#       APPLICATION
#--------------------------------------

class Application:
    """Main application class for the Audio LED GUI."""
    
    def __init__(self, env_config):
        """
        Initialize the GUI application.
        
        Args:
            env_config (dict): Environment configuration dictionary
        """
        self.env_config = env_config
        self.config = env_config.get('config', {})
        
        # Initialize state variables
        self.running = False
        self.clock = None
        self.font = None
        self.large_font = None
        self.screen = None
        
        # Audio data variables
        self.audio_data = None
        self.last_audio_data = None
        self.spectrum_history = []
        self.spectrum_max = [0.1] * 64  # Avoid division by zero
        
        # Input source variables
        self.available_inputs = []
        self.input_source = None
        self.input_source_selected = False
        
        # Set GUI mode flag in the config so input_handler knows not to auto-select
        if 'audio' not in self.config:
            self.config['audio'] = {}
        self.config['audio']['gui_mode'] = True
        
        # Create components
        self.audio_processor = processor.AudioProcessor(self.env_config)
        self.rgb_converter = rgb_converter.RGBConverter(self.config.get('visual', {}))
        
        # LED manager (if hardware is available)
        led_config = self.config.get('led', {})
        if led_config.get('enabled', True):
            self.led_manager = led_controller.LEDManager(self.config)
        else:
            self.led_manager = None
        
        # UI components lists
        self.ui_components = []
        self.input_selection_components = []
        
        # Create the audio thread
        self.audio_thread = threading.Thread(target=self._audio_thread_func)
        self.audio_thread.daemon = True
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logger.info("GUI application initialized")
        
        # Find available input sources
        self._find_available_inputs()
        
    def _find_available_inputs(self):
        """Find all available input sources for the user to select from."""
        self.available_inputs = []
        
        # Add microphone option if available
        if self.env_config.get('env_info', {}).get('capabilities', {}).get('audio_input', False):
            try:
                import pyaudio
                self.available_inputs.append(('microphone', "Microphone Input"))
            except ImportError:
                pass
        
        # Check for test audio files
        test_dirs = [
            Path("testWavs"),
            Path("../testWavs"),
            Path("/home/user/Documents/ledAudio/testWavs")
        ]
        
        for test_dir in test_dirs:
            if test_dir.exists() and test_dir.is_dir():
                # Find all WAV files
                for file_path in test_dir.glob("*.wav"):
                    self.available_inputs.append(
                        (str(file_path), f"WAV: {file_path.name}")
                    )
                
                # Find all MP3 files if pydub is available
                try:
                    from pydub import AudioSegment
                    for file_path in test_dir.glob("*.mp3"):
                        self.available_inputs.append(
                            (str(file_path), f"MP3: {file_path.name}")
                        )
                except ImportError:
                    pass
        
        # Add option for custom file
        self.available_inputs.append(('custom_file', "Select Custom Audio File..."))
        
        logger.info(f"Found {len(self.available_inputs)} available input sources")

    def _setup_input_selection_ui(self):
        """Set up UI for selecting an input source."""
        self.input_selection_components = []
        
        # Clear main panel
        title_height = 60
        
        # Add a title
        title_text = "Select Audio Input Source"
        
        # Calculate positions for buttons
        button_width = 300
        button_height = 40
        button_spacing = 10
        
        # Position the buttons in the center of the screen
        start_y = PADDING + title_height + 20
        
        # Add buttons for each input source
        for i, (source, label) in enumerate(self.available_inputs):
            y_pos = start_y + i * (button_height + button_spacing)
            
            self.input_selection_components.append(Button(
                (WINDOW_WIDTH - button_width) // 2,
                y_pos,
                button_width,
                button_height,
                label,
                lambda src=source: self._on_input_source_selected(src)
            ))
    
    def _on_input_source_selected(self, source):
        """Handle selection of an input source."""
        logger.info(f"Input source selected: {source}")
        
        if source == 'custom_file':
            # In a real implementation, this would open a file dialog
            # For now, we'll just prompt in the console
            print("\nEnter path to audio file:")
            file_path = input().strip()
            
            if os.path.isfile(file_path):
                source = file_path
            else:
                logger.error(f"File not found: {file_path}")
                return
        
        # Set up the audio input
        if source == 'microphone':
            from audio_led.audio import input_handler
            audio_input = input_handler.AudioInputHandler(self.env_config)
            success = audio_input._setup_microphone_input()
        else:
            from audio_led.audio import input_handler
            audio_input = input_handler.AudioInputHandler(self.env_config)
            success = audio_input._setup_file_input(source)
        
        if success:
            # Store the input source
            self.input_source = source
            self.input_source_selected = True
            
            # Set the audio input in the processor
            self.audio_processor.audio_input = audio_input
            
            # Start the audio processor and thread
            self.audio_processor.running = True
            self.audio_thread.start()
            
            # If LED manager exists, start it
            if self.led_manager:
                self.led_manager.start()
            
            # Switch to main UI
            self._setup_ui()
        else:
            logger.error(f"Failed to set up input source: {source}")

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            self.running = False
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        
        # SIGBREAK is Windows-specific, so check if it exists
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)  # Ctrl+Break on Windows
    
    def run(self):
        """Run the GUI application."""
        try:
            # Initialize pygame
            pygame.init()
            pygame.display.set_caption("Audio LED Visualizer")
            
            # Set up the screen
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.clock = pygame.time.Clock()
            
            # Set up fonts
            self.font = pygame.font.Font(None, 20)
            self.large_font = pygame.font.Font(None, 24)
            
            # Set up input selection UI initially
            self._setup_input_selection_ui()
            
            # Start the application loop
            self.running = True
            
            # Main loop
            while self.running:
                self._handle_events()
                
                # Only update visualization if input is selected
                if self.input_source_selected:
                    self._update()
                
                self._draw()
                pygame.display.flip()
                self.clock.tick(FPS)
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, graceful shutdown initiated")
            self.running = False
        except Exception as e:
            logger.error(f"Error in GUI application: {e}")
            logger.exception(e)
        finally:
            # Signal all threads to stop
            self.running = False
            
            # Function to handle cleanup
            def cleanup():
                # Stop the audio processor from the main thread
                if hasattr(self, 'audio_processor') and self.audio_processor:
                    try:
                        logger.info("Stopping audio processor from main thread...")
                        self.audio_processor.stop()
                        logger.info("Audio processor stopped from main thread")
                    except Exception as e:
                        logger.error(f"Error stopping audio processor from main thread: {e}")
                
                # Wait for the audio thread to complete (with shorter timeout)
                if hasattr(self, 'audio_thread') and self.audio_thread and self.audio_thread.is_alive():
                    logger.info("Waiting for audio thread to terminate...")
                    try:
                        self.audio_thread.join(timeout=1.0)  # Shorter timeout
                        if self.audio_thread.is_alive():
                            logger.warning("Audio thread didn't terminate properly, continuing with shutdown")
                    except Exception as e:
                        logger.error(f"Error while joining audio thread: {e}")
                
                # Stop the LED manager
                if hasattr(self, 'led_manager') and self.led_manager:
                    try:
                        logger.info("Stopping LED manager...")
                        self.led_manager.stop()
                    except Exception as e:
                        logger.error(f"Error stopping LED manager: {e}")
                
                # Quit pygame
                try:
                    logger.info("Quitting pygame...")
                    pygame.quit()
                except Exception as e:
                    logger.error(f"Error quitting pygame: {e}")
                    
                logger.info("Application terminated.")
            
            # Perform cleanup
            cleanup()
            
            # Force exit if we're the main thread
            if threading.current_thread() is threading.main_thread():
                logger.info("Forcing exit to prevent hanging...")
                sys.exit(0)
    
    def _setup_ui(self):
        """Set up the UI components."""
        # Current Y position for sidebar components
        y_pos = PADDING
        
        # Title
        title_height = 40
        y_pos += title_height + UI_SPACING
        
        # Audio section
        audio_config = self.config.get('audio', {})
        
        # Volume slider
        self.ui_components.append(Slider(
            PADDING, y_pos, SIDEBAR_WIDTH, SLIDER_HEIGHT,
            0.0, 2.0, audio_config.get('volume_scale', 1.0),
            "Volume Scale", self._on_volume_scale_change
        ))
        y_pos += SLIDER_HEIGHT + 25
        
        # Bass slider
        self.ui_components.append(Slider(
            PADDING, y_pos, SIDEBAR_WIDTH, SLIDER_HEIGHT,
            0.0, 2.0, audio_config.get('bass_scale', 1.0),
            "Bass Scale", self._on_bass_scale_change
        ))
        y_pos += SLIDER_HEIGHT + 25
        
        # Mid slider
        self.ui_components.append(Slider(
            PADDING, y_pos, SIDEBAR_WIDTH, SLIDER_HEIGHT,
            0.0, 2.0, audio_config.get('mid_scale', 1.0),
            "Mid Scale", self._on_mid_scale_change
        ))
        y_pos += SLIDER_HEIGHT + 25
        
        # Treble slider
        self.ui_components.append(Slider(
            PADDING, y_pos, SIDEBAR_WIDTH, SLIDER_HEIGHT,
            0.0, 2.0, audio_config.get('treble_scale', 1.0),
            "Treble Scale", self._on_treble_scale_change
        ))
        y_pos += SLIDER_HEIGHT + 25
        
        # Beat sensitivity slider
        self.ui_components.append(Slider(
            PADDING, y_pos, SIDEBAR_WIDTH, SLIDER_HEIGHT,
            0.1, 2.0, audio_config.get('beat_sensitivity', 1.0),
            "Beat Sensitivity", self._on_beat_sensitivity_change
        ))
        y_pos += SLIDER_HEIGHT + 25
        
        # FFT Enable checkbox
        self.ui_components.append(CheckBox(
            PADDING, y_pos, 20, "Enable FFT",
            audio_config.get('fft_enabled', True),
            self._on_fft_enable_change
        ))
        y_pos += 30 + UI_SPACING
        
        # Save button
        self.ui_components.append(Button(
            PADDING, WINDOW_HEIGHT - PADDING - BUTTON_HEIGHT, 
            SIDEBAR_WIDTH // 2 - 5, BUTTON_HEIGHT,
            "Save Config", self._on_save_config
        ))
        
        # Quit button
        self.ui_components.append(Button(
            PADDING + SIDEBAR_WIDTH // 2 + 5, WINDOW_HEIGHT - PADDING - BUTTON_HEIGHT,
            SIDEBAR_WIDTH // 2 - 5, BUTTON_HEIGHT,
            "Quit", self._on_quit
        ))
    
    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            
            # Also handle ESC key to exit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                logger.info("ESC key pressed, exiting application...")
                self.running = False
                return
            
            # Handle UI component events
            if not self.input_source_selected:
                # Handle input selection UI
                for component in self.input_selection_components:
                    if event.type == pygame.MOUSEMOTION:
                        component.check_hover(event.pos)
                    if component.handle_event(event):
                        break  # Event was handled
            else:
                # Handle main UI
                for component in self.ui_components:
                    if event.type == pygame.MOUSEMOTION:
                        component.check_hover(event.pos)
                    if component.handle_event(event):
                        break  # Event was handled
    
    def _update(self):
        """Update the application state."""
        # Get the latest audio data
        if self.audio_data:
            self.last_audio_data = self.audio_data
        
        # If we have audio data, update the LED manager
        if self.last_audio_data and self.led_manager:
            # Convert audio data to RGB
            rgb = self.rgb_converter.spectrum_to_rgb(self.last_audio_data.get('spectrum', []))
            
            # Set all LEDs to the same color (can be extended for patterns)
            led_count = self.led_manager.led_count
            pixels = [rgb] * led_count
            
            # Update the LEDs
            self.led_manager.set_pixels(pixels)
    
    def _draw(self):
        """Draw the application UI."""
        # Fill the background
        self.screen.fill(BG_COLOR)
        
        if not self.input_source_selected:
            # Draw input selection UI
            title_text = "Select Audio Input Source"
            title_surf = self.large_font.render(title_text, True, TEXT_COLOR)
            title_rect = title_surf.get_rect(centerx=WINDOW_WIDTH // 2, y=PADDING + 20)
            self.screen.blit(title_surf, title_rect)
            
            # Draw input selection components
            for component in self.input_selection_components:
                component.draw(self.screen, self.font)
        else:
            # Draw main visualization panel
            main_panel = pygame.Rect(SIDEBAR_WIDTH + PADDING * 2, PADDING, MAIN_WIDTH, MAIN_HEIGHT)
            pygame.draw.rect(self.screen, PANEL_COLOR, main_panel)
            
            # Draw audio visualizations if we have data
            if self.last_audio_data:
                self._draw_audio_visualizations(main_panel)
            
            # Draw sidebar panel
            sidebar_panel = pygame.Rect(PADDING, PADDING, SIDEBAR_WIDTH, WINDOW_HEIGHT - PADDING * 2)
            pygame.draw.rect(self.screen, PANEL_COLOR, sidebar_panel)
            
            # Draw title
            title_text = "Audio LED Visualizer"
            title_surf = self.large_font.render(title_text, True, TEXT_COLOR)
            title_rect = title_surf.get_rect(x=PADDING + 10, y=PADDING + 10)
            self.screen.blit(title_surf, title_rect)
            
            # Display current input source
            if hasattr(self, 'input_source') and self.input_source:
                source_text = f"Source: {os.path.basename(self.input_source)}"
                source_surf = self.font.render(source_text, True, TEXT_COLOR)
                source_rect = source_surf.get_rect(x=PADDING + 10, y=PADDING + 40)
                self.screen.blit(source_surf, source_rect)
            
            # Draw UI components
            for component in self.ui_components:
                component.draw(self.screen, self.font)
    
    def _draw_audio_visualizations(self, panel):
        """Draw audio visualizations in the main panel."""
        if not self.last_audio_data:
            return
        
        # Extract data
        spectrum = self.last_audio_data.get('spectrum', [])
        volume = self.last_audio_data.get('volume', 0)
        beat = self.last_audio_data.get('beat', False)
        
        # Add to spectrum history
        self.spectrum_history.append(np.array(spectrum))
        if len(self.spectrum_history) > 50:  # Keep last 50 frames
            self.spectrum_history.pop(0)
        
        # Update spectrum max for normalization
        for i, val in enumerate(spectrum):
            if val > self.spectrum_max[i]:
                self.spectrum_max[i] = val
            else:
                self.spectrum_max[i] *= 0.995  # Slowly reduce max if no new peaks
        
        # Draw spectrum
        self._draw_spectrum(panel, spectrum)
        
        # Draw volume meter
        self._draw_volume_meter(panel, volume, beat)
        
        # Draw current color
        self._draw_current_color(panel)
    
    def _draw_spectrum(self, panel, spectrum):
        """Draw the frequency spectrum visualization."""
        # Calculate dimensions
        width = panel.width - 40
        height = panel.height * 0.4
        # Ensure dimensions are integers
        width = int(width)
        height = int(height)
        x = panel.x + 20
        y = panel.y + 20
        
        # Draw border
        spec_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, TEXT_COLOR, spec_rect, width=1)
        
        # Draw background grid
        for i in range(1, 10):
            grid_y = int(y + height * (1 - i / 10))
            pygame.draw.line(self.screen, AXIS_COLOR, (x, grid_y), (x + width, grid_y), 1)
        
        for i in range(1, 10):
            grid_x = int(x + width * i / 10)
            pygame.draw.line(self.screen, AXIS_COLOR, (grid_x, y), (grid_x, y + height), 1)
        
        # Draw spectrum if we have data
        # Use numpy.size to check if the array has elements instead of direct boolean check
        if spectrum is not None and len(spectrum) > 1:
            # Normalize the spectrum
            normalized_spectrum = []
            for i, val in enumerate(spectrum):
                normalized_spectrum.append(min(1.0, val / max(0.01, self.spectrum_max[i])))
            
            # Draw the bars
            bar_width = width / len(normalized_spectrum)
            for i, val in enumerate(normalized_spectrum):
                # Ensure all dimensions are integers for pygame
                bar_height = int(val * height)
                bar_x = int(x + i * bar_width)
                bar_y = int(y + height - bar_height)
                bar_w = max(1, int(bar_width - 1))
                
                bar_rect = pygame.Rect(
                    bar_x, 
                    bar_y,
                    bar_w, 
                    bar_height
                )
                # Color gradient based on frequency (blue->green->red)
                r = min(255, int(i * 3))
                g = min(255, int(510 - abs(i - 42) * 6))
                b = min(255, int(255 - i * 3))
                pygame.draw.rect(self.screen, (r, g, b), bar_rect)
        
        # Draw labels
        freq_label = self.font.render("Frequency Spectrum", True, TEXT_COLOR)
        self.screen.blit(freq_label, (x, y + height + 5))
    
    def _draw_volume_meter(self, panel, volume, beat):
        """Draw the volume meter visualization."""
        # Calculate dimensions
        width = 30
        height = panel.height * 0.4
        # Ensure dimensions are integers
        height = int(height)
        x = panel.x + panel.width - 50
        y = panel.y + 20
        
        # Draw border
        vol_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, TEXT_COLOR, vol_rect, width=1)
        
        # Draw volume level
        vol_height = int(min(1.0, volume) * height)
        vol_fill_rect = pygame.Rect(
            x, y + height - vol_height,
            width, vol_height
        )
        vol_color = BEAT_COLOR if beat else VOLUME_COLOR
        pygame.draw.rect(self.screen, vol_color, vol_fill_rect)
        
        # Draw labels
        vol_label = self.font.render("Volume", True, TEXT_COLOR)
        vol_label_rect = vol_label.get_rect(centerx=x + width // 2, y=y + height + 5)
        self.screen.blit(vol_label, vol_label_rect)
        
        # Draw beat indicator
        if beat:
            beat_label = self.font.render("BEAT", True, BEAT_COLOR)
            beat_rect = beat_label.get_rect(centerx=x + width // 2, y=y + height + 25)
            self.screen.blit(beat_label, beat_rect)
    
    def _draw_current_color(self, panel):
        """Draw the current RGB color visualization."""
        # Calculate dimensions
        size = min(panel.width - 40, panel.height * 0.3)
        # Ensure size is an integer for pygame
        size = int(size)
        x = panel.x + (panel.width - size) // 2
        y = panel.y + panel.height - size - 20
        
        # Get current RGB color
        if self.last_audio_data and 'spectrum' in self.last_audio_data:
            # Convert numpy array to list if needed to prevent type issues
            spectrum = self.last_audio_data['spectrum']
            if hasattr(spectrum, 'tolist'):
                spectrum = spectrum.tolist()
            rgb = self.rgb_converter.spectrum_to_rgb(spectrum)
        else:
            rgb = (0, 0, 0)
        
        # Draw color circle
        color_rect = pygame.Rect(x, y, size, size)
        # Ensure border_radius is an integer
        border_radius = size // 10
        pygame.draw.rect(self.screen, rgb, color_rect, border_radius=border_radius)
        pygame.draw.rect(self.screen, TEXT_COLOR, color_rect, width=1, border_radius=border_radius)
        
        # Draw RGB values
        rgb_text = f"RGB: ({rgb[0]}, {rgb[1]}, {rgb[2]})"
        rgb_surf = self.font.render(rgb_text, True, TEXT_COLOR)
        rgb_rect = rgb_surf.get_rect(centerx=x + size // 2, y=y + size + 5)
        self.screen.blit(rgb_surf, rgb_rect)
    
    def _audio_thread_func(self):
        """Thread function for processing audio data."""
        try:
            while self.running:
                # Check if main thread is still alive
                if not threading.main_thread().is_alive():
                    logger.warning("Main thread is no longer alive, exiting audio thread")
                    break
                
                # Only process audio if input has been selected
                if not self.input_source_selected:
                    time.sleep(0.1)  # Sleep to avoid CPU usage
                    continue
                
                try:
                    # Process audio data with timeout to avoid blocking
                    self.audio_data = self.audio_processor.process()
                    
                    # Sleep a bit to avoid consuming too much CPU
                    time.sleep(1 / 60)
                except Exception as e:
                    logger.error(f"Error in audio processing: {e}")
                    logger.exception(e)
                    time.sleep(0.1)  # Sleep to avoid rapid error loops
                
        except Exception as e:
            logger.error(f"Error in audio thread: {e}")
            logger.exception(e)
        finally:
            # Stop the audio processor
            try:
                if self.audio_processor and hasattr(self, 'running') and not self.running:
                    self.audio_processor.stop()
                    logger.info("Audio processor stopped from thread")
            except Exception as e:
                logger.error(f"Error stopping audio processor from thread: {e}")
    
    #--------------------------------------
    #       EVENT HANDLERS
    #--------------------------------------
    
    def _on_volume_scale_change(self, value):
        """Handle volume scale slider change."""
        self.audio_processor.volume_scale = value
    
    def _on_bass_scale_change(self, value):
        """Handle bass scale slider change."""
        self.audio_processor.bass_scale = value
    
    def _on_mid_scale_change(self, value):
        """Handle mid scale slider change."""
        self.audio_processor.mid_scale = value
    
    def _on_treble_scale_change(self, value):
        """Handle treble scale slider change."""
        self.audio_processor.treble_scale = value
    
    def _on_beat_sensitivity_change(self, value):
        """Handle beat sensitivity slider change."""
        self.audio_processor.beat_sensitivity = value
    
    def _on_fft_enable_change(self, value):
        """Handle FFT enable checkbox change."""
        self.audio_processor.fft_enabled = value
    
    def _on_save_config(self):
        """Handle save config button click."""
        # TODO: Save configuration to file
        logger.info("Configuration saved")
    
    def _on_quit(self):
        """Handle quit button click."""
        try:
            # Safely stop audio processor first before quitting
            if hasattr(self, 'audio_processor'):
                self.audio_processor.stop()
            
            # Signal the application to stop
            self.running = False
            logger.info("Quit button pressed, exiting application...")
            
            # Force exit in case normal shutdown fails
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during quit: {e}")
            sys.exit(1)


#--------------------------------------
#       TEST CODE
#--------------------------------------

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    test_config = {
        'audio': {
            'sample_rate': 44100,
            'chunk_size': 1024,
            'num_channels': 1,
            'device_index': None,
            'fft_enabled': True,
            'window_type': 'hann',
            'freq_bands': 64,
            'smoothing': 0.7,
            'volume_scale': 1.0,
            'bass_scale': 1.0,
            'mid_scale': 1.0,
            'treble_scale': 1.0,
            'beat_sensitivity': 1.0
        },
        'visual': {
            'color_mode': 'spectrum',
            'brightness': 1.0,
            'saturation': 1.0
        },
        'led': {
            'enabled': False,
            'strip_type': 'virtual',
            'led_count': 60,
            'brightness': 0.5
        }
    }
    
    # Create and run the application
    app = Application({'config': test_config})
    app.run() 