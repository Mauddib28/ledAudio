# Import for Raspi GPIO    -   For Pico-W
from machine import Pin, PWM
# Import for time
from time import sleep
# Import for OS
import os

# Configuration of the GPIO -   For Pico-W
red_led = Pin(17, mode=Pin.OUT)
green_led = Pin(22, mode=Pin.OUT)
blue_led = Pin(16, mode=Pin.OUT)
# Set the freq for the PWM
pwm_freq = 1000

# Debug Variable
dbg = 0
test_sleep_flag = 1

# Setup for PWM configuration of pins
pwm__red_led = PWM(red_led)
pwm__green_led = PWM(green_led)
pwm__blue_led = PWM(blue_led)
# Set the frequency for the PWM pins
pwm__red_led.freq(pwm_freq)
pwm__green_led.freq(pwm_freq)
pwm__blue_led.freq(pwm_freq)

# Function for Converting 0-255 into an associated Duty Cycle
def setLights(pin, brightness):
    # Will return a value between 65025 to 0 with scoping to try and provide a range
    realBrightness = int(int(brightness) * (float(65025 / 255.0)))
    # Uses the duty_u16 function to provide PWM control on the Pico-W GPIO
    pin.duty_u16(realBrightness)

# Function to create test file
def createTestFile(filename):
    write_file = open(filename, "w")
    # Writing the test file information
    write_file.write("Red\tGreen\tBlue")
    write_file.close()

## Main Code

# Set all the LEDs to off to start
setLights(pwm__red_led, 0)
setLights(pwm__green_led, 0)
setLights(pwm__blue_led, 0)

'''
# Testing change in lights
print("[*] Moving brightness up")
for value in range(0, 255):
    print("[*] Value:\t{0}".format(value))
    setLights(pwm__red_led, value)
    setLights(pwm__green_led, value)
    setLights(pwm__blue_led, value)
    sleep(0.1)
print("[*] Moving brightness down")
sleep(2)
for value in range(255, 0, -1):
    print("[*] Value:\t{0}".format(value))
    setLights(pwm__red_led, value)
    setLights(pwm__green_led, value)
    setLights(pwm__blue_led, value)
    sleep(0.1)
print("[+] Completed Light test")
sleep(2)

setLights(pwm__red_led, 0)
setLights(pwm__green_led, 0)
setLights(pwm__blue_led, 0)
'''

# Check for existence of the conversion file
conversion_filename = "audio-to-rgb.conversion"
#if os.path.isfile(conversion_filename):
try:
    conversion_debugging_input = open(conversion_filename, 'r')
    first_time_completed = 0
    print("[+] Audio conversion file was found")
except:
    print("[-] Unable to find the conversion file")
    print("[!] Open and save the necessary file to the Raspi Pico-W OS")
    
print("[*] Reading line from input file\n")
while True:
    rgb = conversion_debugging_input.readline()
    if not rgb:
        break
    if dbg != 0:
        print("Read Line:\t\t{0}".format(rgb))

    # Evaluating reads from the files
    if rgb == "Red\tGreen\tBlue\n":
        print("... Header file.... Ignoring\n")
    elif len(rgb) == 0:
        if first_time_completed != 1:
            print("[+] Completed Read through Input File\n")
            first_time_completed = 1
    else:
        rgb_parsed = rgb.strip().split('\t')
        try:
            r = rgb_parsed[0]
            g = rgb_parsed[1]
            b = rgb_parsed[2]
        except IndexError:
            print("[!] Error.... Input:\t{0}\n".format(rgb))
        if dbg != 0:
            print("... Debug - RGB:\t{0}\n\tRed:\t{1}\n\tGreen:\t{2}\n\tBlue:\t{3}\n".format(rgb_parsed, r, g, b))
        setLights(pwm__red_led, r)
        setLights(pwm__green_led, g)
        setLights(pwm__blue_led, b)
        if dbg != 0:
            print("... Waiting time {0} seconds before next read\n".format(test_rgb_wait_time_s))
        if test_sleep_flag != 0:
            test_rgb_wait_time_s = 0.0001
            # Note: 0.01 is Aura-causing
            sleep(test_rgb_wait_time_s)

setLights(pwm__red_led, 0)
setLights(pwm__green_led, 0)
setLights(pwm__blue_led, 0)

print("[+] Completed conversion test for Pico W")