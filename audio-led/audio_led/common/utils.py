#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the Audio-LED project

import time
import numpy as np
import logging

# Initialize logger
logger = logging.getLogger(__name__)

def resample_frame(frame, target_length):
    """Resample a frame to a target length
    
    Parameters
    ----------
    frame : numpy.ndarray
        Frame to resample
    target_length : int
        Target length
        
    Returns
    -------
    numpy.ndarray
        Resampled frame
    """
    if frame is None or len(frame) == 0:
        return np.zeros(target_length)
        
    if len(frame) == target_length:
        return frame
        
    # Simple linear interpolation
    indices = np.linspace(0, len(frame) - 1, target_length)
    result = np.zeros(target_length)
    
    for i in range(target_length):
        idx = indices[i]
        idx_floor = int(np.floor(idx))
        idx_ceil = min(idx_floor + 1, len(frame) - 1)
        weight = idx - idx_floor
        result[i] = frame[idx_floor] * (1 - weight) + frame[idx_ceil] * weight
        
    return result

def get_elapsed(start_time):
    """Get elapsed time in seconds
    
    Parameters
    ----------
    start_time : float
        Start time in seconds
        
    Returns
    -------
    float
        Elapsed time in seconds
    """
    return time.time() - start_time

def map_range(value, in_min, in_max, out_min, out_max):
    """Map a value from one range to another
    
    Parameters
    ----------
    value : float
        Value to map
    in_min : float
        Input minimum
    in_max : float
        Input maximum
    out_min : float
        Output minimum
    out_max : float
        Output maximum
        
    Returns
    -------
    float
        Mapped value
    """
    # Ensure we don't divide by zero
    if in_max - in_min == 0:
        return out_min
        
    # Calculate the mapping
    mapped_value = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    return mapped_value

def clamp(value, min_value, max_value):
    """Clamp a value to a range
    
    Parameters
    ----------
    value : float
        Value to clamp
    min_value : float
        Minimum value
    max_value : float
        Maximum value
        
    Returns
    -------
    float
        Clamped value
    """
    return max(min_value, min(value, max_value)) 