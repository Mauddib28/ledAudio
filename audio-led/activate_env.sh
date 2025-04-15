#!/bin/bash

# Activate the Python virtual environment for Audio LED Visualization

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup_env.sh first."
    echo "    ./setup_env.sh"
    exit 1
fi

# Activate the virtual environment
source venv/bin/activate

echo "=====================================================
Audio LED Visualization Environment ACTIVATED

You can now run the audio_led.py script:
    python audio_led.py

To deactivate the environment:
    deactivate
or:
    ./deactivate_env.sh
======================================================"

# Add environment info to the prompt
export PS1="(audio-led) $PS1" 