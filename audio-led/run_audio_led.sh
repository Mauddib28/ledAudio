#!/bin/bash

# Run audio_led.py in the Python virtual environment

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup first..."
    ./setup_env.sh
fi

# Default settings
USE_GUI=false
TIMEOUT=300  # 5 minute default timeout
NO_DISPLAY="--no-display"
INPUT_SOURCE=""
OUTPUT_METHOD=""
HELP_MODE=false

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
                NO_DISPLAY=""
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
                    NO_DISPLAY=""
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

# Activate the virtual environment
source venv/bin/activate

echo "Running Audio LED Visualization..."
echo "Input: $INPUT_SOURCE"
echo "Output: $OUTPUT_METHOD"
echo "Display mode: $( [ -z "$NO_DISPLAY" ] && echo "enabled" || echo "disabled" )"

# Run with appropriate flags
if $USE_GUI; then
    python audio_led.py $INPUT_SOURCE $OUTPUT_METHOD --gui
else
    python audio_led.py $INPUT_SOURCE $OUTPUT_METHOD $NO_DISPLAY --timeout $TIMEOUT
    
    if [ -z "$INPUT_SOURCE" ] || [ -z "$OUTPUT_METHOD" ]; then
        echo ""
        echo "Tip: For future runs, you can specify both input and output directly:"
        echo "  ./run_audio_led.sh -i microphone -o file"
        echo "Run ./run_audio_led.sh --help for more options"
    fi
fi

# Script output status
status=$?

# Deactivate the virtual environment
deactivate

exit $status 