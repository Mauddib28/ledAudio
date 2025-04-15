#!/bin/bash

# Run audio_led.py in the Python virtual environment

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup first..."
    ./setup_env.sh
fi

# Activate the virtual environment, run the script, then deactivate
source venv/bin/activate

echo "Running Audio LED Visualization..."
python audio_led.py "$@"

# Script output status
status=$?

# Deactivate the virtual environment
deactivate

exit $status 