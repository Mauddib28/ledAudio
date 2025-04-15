#!/bin/bash

# Deactivate the Python virtual environment for Audio LED Visualization

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "No active virtual environment detected."
    exit 0
fi

# Deactivate the virtual environment
deactivate

echo "=========================================
Audio LED Visualization Environment DEACTIVATED

To activate the environment again:
    ./activate_env.sh
=========================================" 