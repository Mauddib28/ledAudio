# Expose utility modules
# GUI module is imported on-demand to avoid dependency requirements
from audio_led.utils.logger import setup_logging
from audio_led.utils.installer import check_and_install_dependencies
