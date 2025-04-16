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
            # Open a file dialog within the GUI
            file_path = self._open_file_dialog()
            
            if file_path and os.path.isfile(file_path):
                source = file_path
            else:
                # If file dialog was cancelled or invalid file, return to input selection
                logger.warning("File selection cancelled or invalid file selected")
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
            # Add a user-visible error message in the GUI
            self._show_error_message(f"Failed to use {source} as input source")

    def _open_file_dialog(self):
        """Open a file dialog to select an audio file.
        
        Returns:
            str or None: Path to selected file, or None if cancelled
        """
        try:
            # Try to use tkinter for file dialog (more robust than pygame's)
            import tkinter as tk
            from tkinter import filedialog
            
            # Hide the main tkinter window
            root = tk.Tk()
            root.withdraw()
            
            # Open the file dialog
            file_types = [
                ('Audio Files', '*.wav *.mp3 *.ogg'),
                ('WAV Files', '*.wav'),
                ('MP3 Files', '*.mp3'),
                ('OGG Files', '*.ogg'),
                ('All Files', '*.*')
            ]
            
            file_path = filedialog.askopenfilename(
                title="Select Audio File",
                filetypes=file_types
            )
            
            # Clean up tkinter
            root.destroy()
            
            return file_path if file_path else None
            
        except ImportError:
            logger.warning("tkinter not available, using custom file selection UI")
            return self._show_custom_file_selector()
    
    def _show_custom_file_selector(self):
        """Show a custom file selector UI built with pygame.
        
        Returns:
            str or None: Path to selected file, or None if cancelled
        """
        # Save the current screen state to restore later
        screen_backup = self.screen.copy()
        
        # Create a file browser UI
        current_dir = os.path.expanduser("~")  # Start in home directory
        selected_file = None
        
        # Font settings
        font = pygame.font.Font(None, 22)
        title_font = pygame.font.Font(None, 28)
        
        # Colors
        bg_color = (50, 50, 60)
        panel_color = (30, 30, 40)
        selected_color = (70, 100, 150)
        dir_color = (70, 130, 200)
        
        # Create directory navigation history
        dir_history = [current_dir]
        history_pos = 0
        
        # File filter
        file_extensions = ['.wav', '.mp3', '.ogg']
        
        # Scroll position for directory listing
        scroll_offset = 0
        max_visible_items = 15
        
        # Add navigation buttons
        up_button_rect = pygame.Rect(50, 60, 30, 30)
        home_button_rect = pygame.Rect(90, 60, 30, 30)
        refresh_button_rect = pygame.Rect(130, 60, 30, 30)
        
        # Additional state variables
        item_height = 30
        list_start_y = 100
        
        while True:
            # Get mouse position for hover effects
            mouse_pos = pygame.mouse.get_pos()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # Cancel and return to previous screen
                        self.screen.blit(screen_backup, (0, 0))
                        return None
                    
                    if event.key == pygame.K_BACKSPACE:
                        # Go up one directory
                        parent_dir = os.path.dirname(current_dir)
                        if parent_dir != current_dir:  # Prevent going above root
                            current_dir = parent_dir
                            dir_history = dir_history[:history_pos+1]
                            dir_history.append(current_dir)
                            history_pos = len(dir_history) - 1
                            scroll_offset = 0  # Reset scroll position
                            selected_file = None
                    
                    if event.key == pygame.K_LEFT and history_pos > 0:
                        # Navigate back in history
                        history_pos -= 1
                        current_dir = dir_history[history_pos]
                        scroll_offset = 0
                        selected_file = None
                    
                    if event.key == pygame.K_RIGHT and history_pos < len(dir_history) - 1:
                        # Navigate forward in history
                        history_pos += 1
                        current_dir = dir_history[history_pos]
                        scroll_offset = 0
                        selected_file = None
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Check navigation buttons
                    if up_button_rect.collidepoint(mouse_pos):
                        # Go up one directory
                        parent_dir = os.path.dirname(current_dir)
                        if parent_dir != current_dir:  # Prevent going above root
                            current_dir = parent_dir
                            dir_history = dir_history[:history_pos+1]
                            dir_history.append(current_dir)
                            history_pos = len(dir_history) - 1
                            scroll_offset = 0  # Reset scroll position
                            selected_file = None
                            continue
                    
                    if home_button_rect.collidepoint(mouse_pos):
                        # Go to home directory
                        current_dir = os.path.expanduser("~")
                        dir_history = dir_history[:history_pos+1]
                        dir_history.append(current_dir)
                        history_pos = len(dir_history) - 1
                        scroll_offset = 0
                        selected_file = None
                        continue
                    
                    if refresh_button_rect.collidepoint(mouse_pos):
                        # Refresh current directory
                        scroll_offset = 0
                        continue
                    
                    # Check if cancel button was clicked
                    cancel_rect = pygame.Rect(WINDOW_WIDTH - 150, WINDOW_HEIGHT - 50, 130, 40)
                    if cancel_rect.collidepoint(mouse_pos):
                        self.screen.blit(screen_backup, (0, 0))
                        return None
                    
                    # Check if select button was clicked
                    select_rect = pygame.Rect(WINDOW_WIDTH - 300, WINDOW_HEIGHT - 50, 130, 40)
                    if select_rect.collidepoint(mouse_pos) and selected_file:
                        self.screen.blit(screen_backup, (0, 0))
                        return selected_file
                    
                    # Scroll handling
                    if event.button == 4:  # Scroll up
                        scroll_offset = max(0, scroll_offset - 1)
                    elif event.button == 5:  # Scroll down
                        # Will be limited by the actual item count in the draw loop
                        scroll_offset += 1
                    
                    # Check if a file or directory in the list was clicked
                    try:
                        entries = sorted(os.listdir(current_dir))
                        dirs = [d for d in entries if os.path.isdir(os.path.join(current_dir, d))]
                        files = [f for f in entries if os.path.isfile(os.path.join(current_dir, f)) 
                                and os.path.splitext(f)[1].lower() in file_extensions]
                        
                        all_items = [('..', True)] + [(d, True) for d in dirs] + [(f, False) for f in files]
                        
                        # Limit scroll offset
                        max_scroll = max(0, len(all_items) - max_visible_items)
                        scroll_offset = min(scroll_offset, max_scroll)
                        
                        # Only check visible items
                        visible_items = all_items[scroll_offset:scroll_offset + max_visible_items]
                        
                        for i, (name, is_dir) in enumerate(visible_items):
                            # Position in the visible list
                            item_rect = pygame.Rect(50, list_start_y + i * item_height, WINDOW_WIDTH - 100, item_height)
                            if item_rect.collidepoint(mouse_pos):
                                if is_dir:
                                    # Navigate into directory
                                    if name == '..':
                                        # Go up one directory
                                        parent_dir = os.path.dirname(current_dir)
                                        if parent_dir != current_dir:  # Prevent going above root
                                            current_dir = parent_dir
                                    else:
                                        # Enter the selected directory
                                        new_dir = os.path.join(current_dir, name)
                                        current_dir = new_dir
                                    
                                    # Update history
                                    dir_history = dir_history[:history_pos+1]
                                    dir_history.append(current_dir)
                                    history_pos = len(dir_history) - 1
                                    
                                    # Reset scroll position
                                    scroll_offset = 0
                                    
                                    # Clear selected file when changing directories
                                    selected_file = None
                                else:
                                    # Select file
                                    selected_file = os.path.join(current_dir, name)
                    except (PermissionError, OSError) as e:
                        logger.error(f"Error accessing directory: {e}")
                        # Don't change directory if we can't access it
                        # Instead, show an error message
            
            # Draw the file browser
            self.screen.fill(bg_color)
            
            # Draw title
            title = "Select Audio File"
            title_surf = title_font.render(title, True, TEXT_COLOR)
            self.screen.blit(title_surf, (WINDOW_WIDTH//2 - title_surf.get_width()//2, 20))
            
            # Draw current directory path
            max_path_width = WINDOW_WIDTH - 200
            path_text = current_dir
            if font.size(path_text)[0] > max_path_width:
                # Truncate path if too long
                path_parts = path_text.split(os.sep)
                truncated_path = os.sep.join(['...'] + path_parts[-3:])
                path_text = truncated_path
                
            dir_surf = font.render(f"Directory: {path_text}", True, TEXT_COLOR)
            self.screen.blit(dir_surf, (170, 65))
            
            # Draw navigation buttons
            # Up button
            pygame.draw.rect(self.screen, dir_color if up_button_rect.collidepoint(mouse_pos) else PANEL_COLOR, 
                            up_button_rect, border_radius=5)
            pygame.draw.rect(self.screen, TEXT_COLOR, up_button_rect, width=1, border_radius=5)
            up_text = font.render("↑", True, TEXT_COLOR)
            self.screen.blit(up_text, (up_button_rect.centerx - up_text.get_width()//2, 
                                    up_button_rect.centery - up_text.get_height()//2))
            
            # Home button
            pygame.draw.rect(self.screen, dir_color if home_button_rect.collidepoint(mouse_pos) else PANEL_COLOR, 
                            home_button_rect, border_radius=5)
            pygame.draw.rect(self.screen, TEXT_COLOR, home_button_rect, width=1, border_radius=5)
            home_text = font.render("🏠", True, TEXT_COLOR)
            self.screen.blit(home_text, (home_button_rect.centerx - home_text.get_width()//2, 
                                        home_button_rect.centery - home_text.get_height()//2))
            
            # Refresh button
            pygame.draw.rect(self.screen, dir_color if refresh_button_rect.collidepoint(mouse_pos) else PANEL_COLOR, 
                            refresh_button_rect, border_radius=5)
            pygame.draw.rect(self.screen, TEXT_COLOR, refresh_button_rect, width=1, border_radius=5)
            refresh_text = font.render("⟳", True, TEXT_COLOR)
            self.screen.blit(refresh_text, (refresh_button_rect.centerx - refresh_text.get_width()//2, 
                                        refresh_button_rect.centery - refresh_text.get_height()//2))
            
            # Draw file panel background
            panel_rect = pygame.Rect(40, 90, WINDOW_WIDTH - 80, WINDOW_HEIGHT - 150)
            pygame.draw.rect(self.screen, panel_color, panel_rect, border_radius=5)
            
            # List files and directories
            try:
                entries = sorted(os.listdir(current_dir))
                dirs = [d for d in entries if os.path.isdir(os.path.join(current_dir, d))]
                files = [f for f in entries if os.path.isfile(os.path.join(current_dir, f)) 
                        and os.path.splitext(f)[1].lower() in file_extensions]
                
                all_items = [('..', True)] + [(d, True) for d in dirs] + [(f, False) for f in files]
                
                # Limit scroll offset based on actual number of items
                max_scroll = max(0, len(all_items) - max_visible_items)
                scroll_offset = min(scroll_offset, max_scroll)
                
                # Draw scrolling file list
                visible_items = all_items[scroll_offset:scroll_offset + max_visible_items]
                
                # Draw scroll indicators if needed
                if scroll_offset > 0:
                    pygame.draw.polygon(self.screen, TEXT_COLOR, [
                        (WINDOW_WIDTH//2 - 10, list_start_y - 15),
                        (WINDOW_WIDTH//2 + 10, list_start_y - 15),
                        (WINDOW_WIDTH//2, list_start_y - 5)
                    ])
                
                if scroll_offset < max_scroll:
                    bottom_y = list_start_y + max_visible_items * item_height + 5
                    pygame.draw.polygon(self.screen, TEXT_COLOR, [
                        (WINDOW_WIDTH//2 - 10, bottom_y),
                        (WINDOW_WIDTH//2 + 10, bottom_y),
                        (WINDOW_WIDTH//2, bottom_y + 10)
                    ])
                
                for i, (name, is_dir) in enumerate(visible_items):
                    # Background for selected item or hover
                    item_rect = pygame.Rect(50, list_start_y + i * item_height, WINDOW_WIDTH - 100, item_height)
                    
                    # Highlight the selected file or hovered item
                    if not is_dir and selected_file == os.path.join(current_dir, name):
                        pygame.draw.rect(self.screen, selected_color, item_rect, border_radius=3)
                    elif item_rect.collidepoint(mouse_pos):
                        pygame.draw.rect(self.screen, (60, 60, 70), item_rect, border_radius=3)
                    
                    # Icon based on type
                    icon = "📁 " if is_dir else "🎵 "
                    if name == "..":
                        icon = "⬆️ "
                        name = "Parent Directory"
                    
                    item_text = icon + name
                    
                    # Text color based on type
                    text_color = (180, 200, 255) if is_dir else TEXT_COLOR
                    
                    item_surf = font.render(item_text, True, text_color)
                    self.screen.blit(item_surf, (item_rect.x + 5, item_rect.y + 5))
            
            except (PermissionError, OSError) as e:
                error_msg = f"Error accessing directory: {str(e)}"
                error_surf = font.render(error_msg, True, (255, 100, 100))
                self.screen.blit(error_surf, (50, 100))
                
                # Add a suggestion to try a different directory
                suggestion = "Try going back to your home directory"
                suggestion_surf = font.render(suggestion, True, (220, 220, 100))
                self.screen.blit(suggestion_surf, (50, 130))
            
            # Draw buttons
            cancel_rect = pygame.Rect(WINDOW_WIDTH - 150, WINDOW_HEIGHT - 50, 130, 40)
            pygame.draw.rect(self.screen, PANEL_COLOR, cancel_rect, border_radius=5)
            pygame.draw.rect(self.screen, TEXT_COLOR, cancel_rect, width=1, border_radius=5)
            
            cancel_text = font.render("Cancel", True, TEXT_COLOR)
            self.screen.blit(cancel_text, (cancel_rect.centerx - cancel_text.get_width()//2, 
                                        cancel_rect.centery - cancel_text.get_height()//2))
            
            select_rect = pygame.Rect(WINDOW_WIDTH - 300, WINDOW_HEIGHT - 50, 130, 40)
            select_color = HIGHLIGHT_COLOR if selected_file else PANEL_COLOR
            pygame.draw.rect(self.screen, select_color, select_rect, border_radius=5)
            pygame.draw.rect(self.screen, TEXT_COLOR, select_rect, width=1, border_radius=5)
            
            select_text = font.render("Select", True, TEXT_COLOR)
            self.screen.blit(select_text, (select_rect.centerx - select_text.get_width()//2, 
                                        select_rect.centery - select_text.get_height()//2))
            
            # Draw key shortcuts help
            help_text = "ESC: Cancel | ←→: History | Scroll to navigate"
            help_surf = font.render(help_text, True, (150, 150, 150))
            self.screen.blit(help_surf, (50, WINDOW_HEIGHT - 30))
            
            pygame.display.flip()
            self.clock.tick(30)
    
    def _show_error_message(self, message):
        """Show an error message in the GUI.
        
        Args:
            message (str): Error message to display
        """
        # Save the current screen state to restore later
        screen_backup = self.screen.copy()
        
        # Font for the message
        font = pygame.font.Font(None, 24)
        
        # Create message box
        box_width = 400
        box_height = 150
        box_x = (WINDOW_WIDTH - box_width) // 2
        box_y = (WINDOW_HEIGHT - box_height) // 2
        
        # Colors
        bg_color = (80, 30, 30)
        border_color = (200, 80, 80)
        
        # Wait for user acknowledgment
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
            
            # Draw message box
            pygame.draw.rect(self.screen, bg_color, (box_x, box_y, box_width, box_height), border_radius=10)
            pygame.draw.rect(self.screen, border_color, (box_x, box_y, box_width, box_height), width=2, border_radius=10)
            
            # Draw message
            lines = message.split('\n')
            for i, line in enumerate(lines):
                text_surf = font.render(line, True, TEXT_COLOR)
                self.screen.blit(text_surf, 
                                (box_x + (box_width - text_surf.get_width()) // 2, 
                                 box_y + 40 + i * 30))
            
            # Draw instruction
            help_text = "Click or press any key to continue"
            help_surf = font.render(help_text, True, (200, 200, 200))
            self.screen.blit(help_surf, 
                          (box_x + (box_width - help_surf.get_width()) // 2, 
                           box_y + box_height - 30))
            
            pygame.display.flip()
            self.clock.tick(30)
        
        # Restore the screen
        self.screen.blit(screen_backup, (0, 0))

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