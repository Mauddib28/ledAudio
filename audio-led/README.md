# Audio LED Visualization System

A modular system for visualizing audio input through LED light patterns. The system processes audio data in real-time, extracts features such as frequency and amplitude, and translates these features into visually appealing LED light patterns.

## Features

- **Real-time audio processing** - Analyzes audio input in real-time to extract frequency spectrum, volume, and beats
- **Dynamic RGB color mapping** - Converts audio features to colorful RGB values that respond to the audio
- **Multiple visualization modes** - Including spectrum, intensity, and colorful modes
- **Hardware abstraction** - Supports different LED types (WS2812B, NeoPixels, APA102, etc.)
- **Platform independent** - Works on desktop (Linux, macOS, Windows) and embedded systems (Raspberry Pi, Pico W)
- **Modular design** - Easy to extend and customize

## Requirements

- Python 3.6+
- NumPy
- (Optional) PyAudio or SoundDevice for microphone input
- (Optional) Various LED libraries (NeoPixel, DotStar, etc.) for hardware control

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-led.git
cd audio-led
```

2. Install dependencies:
```bash
pip install numpy
```

3. Optional dependencies for advanced features:
```bash
# For microphone input
pip install pyaudio

# For MP3 support
pip install pydub

# For GUI visualization
pip install pygame

# For LED hardware control (on Raspberry Pi)
pip install adafruit-circuitpython-neopixel
```

## Usage

### Basic usage:

```bash
python audio_led.py
```

The system will automatically detect available audio sources and LED devices, selecting the most appropriate ones.

### Command-line options:

```bash
python audio_led.py -i <input_source> -o <output_method> -c <config_file> -d -g
```

- `-i, --input`: Specify audio input source (microphone, file path)
- `-o, --output`: Specify output method (led_pwm, led_strip, display, file, none)
- `-c, --config`: Path to configuration file
- `-d, --debug`: Enable debug logging
- `-g, --gui`: Launch with graphical user interface
- `--no-display`: Force headless mode with no GUI windows (uses file output)
- `-t, --timeout`: Maximum runtime in seconds (default: 3600s/1hr)
- `--detect`: Detect and list available audio input/output devices

### Running the application:

The easiest way to run the application is with the provided scripts:

```bash
# Setup the environment (first time)
./setup_env.sh

# Run with required input and output arguments
./run_audio_led.sh -i microphone -o file

# Using a file as input with display output
./run_audio_led.sh -i path/to/music.wav -o display

# Run with GUI
./run_audio_led.sh --gui

# Run with custom timeout (in seconds)
./run_audio_led.sh -i microphone -o file --timeout=600

# Show all available options
./run_audio_led.sh --help
```

If you don't provide the required input and output arguments, you will be prompted to select them through an interactive interface.

### Visualization examples:

To visualize the output in the terminal:
```bash
./display_results.sh
```

## Configuration

You can create a custom `config.yaml` file to configure the system:

```yaml
audio:
  input_source: default
  sample_rate: 44100
  chunk_size: 1024

processing:
  fft_enabled: true
  window_type: "blackman"
  bands: 64
  smoothing: 0.5

visual:
  output_method: display  # Use 'display' for visualization, 'leds' for actual LEDs
  color_mode: colorful
  brightness: 255
  led_count: 60

hardware:
  led_type: "auto"
  led_count: 60
  led_pin: 18
```

## Adding New Hardware

The system's modular design makes it easy to add support for new hardware:

1. Create a new controller class in the appropriate module
2. Implement the required interface methods
3. Register the new controller in the device manager

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on original work from:
  - musicReact.py by David Ordnung and Paul A. Wortman
  - psynesthesia_modified.py by rho-bit 

## Additional Information:
