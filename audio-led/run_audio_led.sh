#!/bin/bash

# Run audio_led.py with or without a Python virtual environment

# Default settings
USE_GUI=false
TIMEOUT=300  # 5 minute default timeout
NO_DISPLAY="--no-display"  # Default to no display
INPUT_SOURCE=""
OUTPUT_METHOD=""
HELP_MODE=false
USE_VENV=true

# Display help function
show_help() {
    echo "Audio LED Visualization System"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -g, --gui                  Enable GUI mode"
    echo "  -i, --input SOURCE         Specify input source (microphone, file.wav)"
    echo "  -o, --output METHOD        Specify output method (led_pwm, led_strip, display, file, none)"
    echo "  --timeout=SECONDS          Set custom timeout (default: 300 seconds)"
    echo "  --display                  Allow display output in headless mode"
    echo "  --no-venv                  Run without using virtual environment"
    echo "  --install-deps             Install missing dependencies"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -i microphone -o file   Use microphone input with file output"
    echo "  $0 -i music.wav -o display Use music.wav as input with display output"
    echo "  $0 -g                      Run in GUI mode"
    echo ""
    echo "If input or output is not specified, you will be prompted to select them."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gui)
            USE_GUI=true
            NO_DISPLAY=""  # Don't use no-display when GUI is requested
            shift
            ;;
        -i=*|--input=*)
            INPUT_SOURCE="--input ${1#*=}"
            shift
            ;;
        -o=*|--output=*)
            OUTPUT_METHOD="--output ${1#*=}"
            # If output method is display, don't force --no-display
            if [[ "${1#*=}" == "display" ]]; then
                NO_DISPLAY=""  # Clear the no-display flag
            fi
            shift
            ;;
        -i|--input)
            if [[ -n "$2" && "$2" != -* ]]; then
                INPUT_SOURCE="--input $2"
                shift 2
            else
                echo "Error: Input source requires an argument"
                exit 1
            fi
            ;;
        -o|--output)
            if [[ -n "$2" && "$2" != -* ]]; then
                OUTPUT_METHOD="--output $2"
                # If output method is display, don't force --no-display
                if [[ "$2" == "display" ]]; then
                    NO_DISPLAY=""  # Clear the no-display flag
                fi
                shift 2
            else
                echo "Error: Output method requires an argument"
                exit 1
            fi
            ;;
        --timeout=*)
            TIMEOUT="${1#*=}"
            shift
            ;;
        --display)
            NO_DISPLAY=""  # Allow user to explicitly enable display
            shift
            ;;
        --no-venv)
            USE_VENV=false
            shift
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        -h|--help)
            HELP_MODE=true
            shift
            ;;
        *)
            # Unknown option
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested
if $HELP_MODE; then
    show_help
fi

# Extract the output method from OUTPUT_METHOD for checking
# For example, "--output display" becomes just "display"
OUTPUT_METHOD_VALUE=""
if [[ $OUTPUT_METHOD == "--output "* ]]; then
    OUTPUT_METHOD_VALUE="${OUTPUT_METHOD#--output }"
fi

# If we're using display output, ensure --no-display is not set
if [[ "$OUTPUT_METHOD_VALUE" == "display" ]]; then
    NO_DISPLAY=""
    echo "Display output mode selected, enabling display functionality"
    
    # Ensure DISPLAY environment variable is set for GUI/display output
    if [ -z "$DISPLAY" ]; then
        echo "DISPLAY environment variable not set, setting to :0"
        export DISPLAY=:0
    fi
    
    # Check if X server is running and accessible
    if ! xdpyinfo >/dev/null 2>&1; then
        echo "Warning: X server not accessible. Display output may not work."
        echo "If running in a headless environment without X server, consider using --no-display or -o file"
    fi
fi

# Check for required dependencies
check_dependencies() {
    if ! python3 -c "import numpy" 2>/dev/null; then
        echo "NumPy not found. This is required."
        MISSING_DEPS=true
    fi
    
    if ! python3 -c "import pyaudio" 2>/dev/null; then
        echo "PyAudio not found. This is required for audio processing."
        MISSING_DEPS=true
    fi
    
    # Check for optional dependencies with warnings
    if ! python3 -c "import soundfile" 2>/dev/null; then
        echo "Warning: soundfile module not found. Audio file handling will use fallback methods."
        echo "You can install it with: pip install soundfile"
    fi
    
    if ! python3 -c "import librosa" 2>/dev/null; then
        echo "Warning: librosa module not found. Audio resampling will use simpler methods."
        echo "You can install it with: pip install librosa"
    fi
    
    # Check for command-line tools
    if ! command -v ffmpeg &> /dev/null; then
        echo "Warning: ffmpeg not found. This is needed for some audio formats."
        echo "Install with: sudo apt install ffmpeg (Ubuntu/Debian) or sudo pacman -S ffmpeg (Arch)"
    fi
}

# Install dependencies if requested
install_dependencies() {
    echo "Installing dependencies..."
    
    # Determine package manager
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        echo "Detected Debian/Ubuntu-based system"
        echo "Installing system dependencies..."
        sudo apt-get update
        sudo apt-get install -y python3-pip python3-numpy python3-pyaudio ffmpeg
        echo "Installing Python packages..."
        pip3 install --user pydub soundfile scipy librosa
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        echo "Detected Arch Linux-based system"
        echo "Installing system dependencies..."
        sudo pacman -Sy --noconfirm python-pip python-numpy python-pyaudio ffmpeg
        echo "Installing Python packages..."
        pip install --user pydub soundfile scipy librosa
    else
        echo "Could not detect package manager. Please install dependencies manually:"
        echo "- python3, python3-pip, numpy, pyaudio, ffmpeg"
        echo "- pydub, soundfile, scipy, librosa (via pip)"
        return 1
    fi
    
    echo "Dependencies installed."
    return 0
}

# Install dependencies if requested
if [ "$INSTALL_DEPS" = true ]; then
    install_dependencies
    if [ $? -ne 0 ]; then
        echo "Failed to install some dependencies."
    fi
fi

# Handle virtual environment
if $USE_VENV; then
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Virtual environment not found."
        echo "Trying to create it automatically..."
        
        # Check if python3-venv is available
        if ! command -v python3 -m venv &> /dev/null; then
            echo "python3-venv not available. Trying to create environment anyway..."
        fi
        
        # Try to create the environment
        python3 -m venv venv 2>/dev/null
        
        if [ $? -ne 0 ]; then
            echo "Failed to create virtual environment."
            echo "You may need to install the python3-venv package:"
            echo "  Ubuntu/Debian: sudo apt install python3-venv"
            echo "  Arch Linux: sudo pacman -S python-virtualenv"
            echo ""
            echo "Running without virtual environment instead..."
            USE_VENV=false
        else
            echo "Virtual environment created."
            # Install basic dependencies in the new environment
            source venv/bin/activate
            pip install --upgrade pip
            pip install numpy pyyaml pyaudio pydub
            # Try to install optional dependencies
            pip install soundfile scipy librosa 2>/dev/null
            deactivate
        fi
    fi
    
    # Activate virtual environment if it exists
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "Virtual environment activation script not found."
        echo "Running without virtual environment..."
        USE_VENV=false
    fi
else
    echo "Running without virtual environment as requested."
    # Check system dependencies instead
    check_dependencies
fi

echo "Running Audio LED Visualization..."
echo "Input: $INPUT_SOURCE"
echo "Output: $OUTPUT_METHOD"
echo "Display mode: $( [ -z "$NO_DISPLAY" ] && echo "enabled" || echo "disabled" )"
echo "Using virtual environment: $( $USE_VENV && echo "yes" || echo "no" )"

# Run with appropriate flags
if $USE_GUI; then
    echo "Starting in GUI mode..."
    # Make sure DISPLAY is set
    if [ -z "$DISPLAY" ]; then
        echo "DISPLAY environment variable not set, setting to :0"
        export DISPLAY=:0
    fi
    python3 audio_led.py $INPUT_SOURCE $OUTPUT_METHOD --gui
else
    python3 audio_led.py $INPUT_SOURCE $OUTPUT_METHOD $NO_DISPLAY --timeout $TIMEOUT
    
    if [ -z "$INPUT_SOURCE" ] || [ -z "$OUTPUT_METHOD" ]; then
        echo ""
        echo "Tip: For future runs, you can specify both input and output directly:"
        echo "  ./run_audio_led.sh -i microphone -o file"
        echo "Run ./run_audio_led.sh --help for more options"
    fi
fi

# Script output status
status=$?

# Deactivate the virtual environment if we used it
if $USE_VENV && [ -n "$VIRTUAL_ENV" ]; then
    deactivate || true
fi

exit $status 