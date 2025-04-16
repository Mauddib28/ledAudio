#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is part of the Audio-LED project

import threading
import logging
import time

# Initialize logger
logger = logging.getLogger(__name__)

class ThreadManager:
    """Thread manager for Audio-LED project
    
    This class provides a simple way to manage threads and ensure
    proper cleanup when the application exits.
    """
    
    # Class-level thread dictionary
    _threads = {}
    _lock = threading.Lock()
    
    @classmethod
    def register_thread(cls, thread_id, thread, is_daemon=True):
        """Register a thread with the manager
        
        Parameters
        ----------
        thread_id : str
            Unique identifier for the thread
        thread : threading.Thread
            Thread object to register
        is_daemon : bool, optional
            Whether the thread should be a daemon, by default True
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        with cls._lock:
            if thread_id in cls._threads:
                logger.warning(f"Thread {thread_id} already registered")
                return False
                
            # Set daemon property
            thread.daemon = is_daemon
            
            # Store the thread
            cls._threads[thread_id] = thread
            
            logger.debug(f"Registered thread: {thread_id}")
            return True
    
    @classmethod
    def unregister_thread(cls, thread_id):
        """Unregister a thread from the manager
        
        Parameters
        ----------
        thread_id : str
            Unique identifier for the thread
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        with cls._lock:
            if thread_id not in cls._threads:
                logger.warning(f"Thread {thread_id} not registered")
                return False
                
            thread = cls._threads.pop(thread_id)
            logger.debug(f"Unregistered thread: {thread_id}")
            return True
    
    @classmethod
    def get_thread(cls, thread_id):
        """Get a thread by ID
        
        Parameters
        ----------
        thread_id : str
            Unique identifier for the thread
            
        Returns
        -------
        threading.Thread
            Thread object, or None if not found
        """
        with cls._lock:
            return cls._threads.get(thread_id)
    
    @classmethod
    def start_thread(cls, thread_id, target, args=(), kwargs={}, daemon=True):
        """Create and start a new thread
        
        Parameters
        ----------
        thread_id : str
            Unique identifier for the thread
        target : callable
            Target function for the thread
        args : tuple, optional
            Arguments for the target function, by default ()
        kwargs : dict, optional
            Keyword arguments for the target function, by default {}
        daemon : bool, optional
            Whether the thread should be a daemon, by default True
            
        Returns
        -------
        threading.Thread
            Thread object, or None if failed
        """
        # Create the thread
        thread = threading.Thread(
            target=target,
            args=args,
            kwargs=kwargs,
            daemon=daemon
        )
        
        # Register the thread
        if cls.register_thread(thread_id, thread, daemon):
            # Start the thread
            try:
                thread.start()
                logger.debug(f"Started thread: {thread_id}")
                return thread
            except Exception as e:
                logger.error(f"Error starting thread {thread_id}: {e}")
                cls.unregister_thread(thread_id)
                return None
        
        return None
    
    @classmethod
    def stop_thread(cls, thread_id, join_timeout=1.0):
        """Stop a thread by ID
        
        This method depends on the thread having a stop mechanism,
        typically a stop event or flag.
        
        Parameters
        ----------
        thread_id : str
            Unique identifier for the thread
        join_timeout : float, optional
            Timeout for joining the thread, by default 1.0
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        with cls._lock:
            thread = cls._threads.get(thread_id)
            
            if not thread:
                logger.warning(f"Thread {thread_id} not found")
                return False
                
            # Try to join the thread
            if thread.is_alive():
                logger.debug(f"Joining thread: {thread_id}")
                thread.join(timeout=join_timeout)
                
            # Unregister the thread
            cls.unregister_thread(thread_id)
            return True
    
    @classmethod
    def cleanup(cls):
        """Clean up all threads registered with the manager
        
        Returns
        -------
        int
            Number of threads cleaned up
        """
        with cls._lock:
            thread_ids = list(cls._threads.keys())
            
        cleaned = 0
        for thread_id in thread_ids:
            if cls.stop_thread(thread_id):
                cleaned += 1
                
        logger.debug(f"Cleaned up {cleaned} threads")
        return cleaned 