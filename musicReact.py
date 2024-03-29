#!/usr/bin/python
# -*- coding: utf-8 -*-

####
# Editted Last on:      2023-07-08
# Editted Last by:      Paul A. Wortman
####

#--------------------------------------
#   BACKGROUND RESEARCH HISTORY
#--------------------------------------

# Nota Bene: This code is a mashed version fo two previous files (fading.py + sampleVolume.py)

 # -----------------------------------------------------
 # File        fading.py
 # Authors     David Ordnung
 # License     GPLv3
 # Web         http://dordnung.de/raspberrypi-ledstrip/
 # -----------------------------------------------------
 # 
 # Copyright (C) 2014-2017 David Ordnung
 # 
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # any later version.
 #  
 # This program is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 # 
 # You should have received a copy of the GNU General Public License
 # along with this program. If not, see <http://www.gnu.org/licenses/>
#

#--------------------------------------
#       RESEARCH / RESOURCES
#--------------------------------------

# This script needs running pigpio (http://abyz.co.uk/rpi/pigpio/)
# Good example of threads (https://python-course.eu/applications-python/pipes-in-python.php)

## This is an example of a simple sound capture script.
##
## The script opens an ALSA pcm for sound capture. Set
## various attributes of the capture, and reads in a loop,
## Then prints the volume.
##
## To test it out, run it and shour at your microphone (input):
##  - This worked

## Pulling Audio-to-Color code from Python 2 Github project (Psynesthesia)
##  - URL:      https://github.com/rho-bit/Psynesthesia
## Basically solves the audio-to-color problem
## 
## Note: Need to streamline and minimize graphics
##  - Minial resource usage to be able to run on a Raspi
##  - Dirty 2to3 conversion of code; TODO: Clean up code and add documentation
##
## Good Resouses:
##  - https://scipython.com/blog/converting-a-spectrum-to-a-colour/
##  - http://www.noah.org/wiki/Wavelength_to_RGB_in_Python

## Can use the fake-rpigpio pypi package to emulate the Raspi while working
##  in non-Raspi environments
##  - URL:      https://pypi.org/project/fake-rpigpio/


###### CONFIGURE THIS ######

# The Pins. Use Broadcom numbers.
RED_PIN   = 17
GREEN_PIN = 22
BLUE_PIN  = 24

# Number of color changes per step (more is faster, less is slower).
# You also can use 0.X floats.
STEPS     	= 50
BRIGHT_STEPS	= 5	# Original code default is 1

# Time given between reads from te test conversion file             [   PURPOSE IS FOR TESTING  ]
test_rgb_wait_time_s    = 0.05

###### END ######

# ----------------
#  Debug bit set for outputing debug messages
# ----------------
dbg = 0
using_microphone_flag = 0
test_sleep_flag = 0

# Bit used for tracking if running on a Raspi or not
raspi_os = None
pico_w_os = None

# ----------------
#  Configuration and Imports for Base OS being Run on (e.g. Raspi B, Pico W, Unix)
# ----------------

# Check to see what kind of system the code is running on
import os
if os.uname()[4][:3] == 'arm':
    raspi_os = 1
    print("[+] Running on Raspi OS (ARM)")
    # Test to see if pi-gpio library is accessible
    try:
        import pigpio
        print("[+] Imported the pigpio library")
    except:
        print("[-] Unable to import the pigpio library")
elif "Pico W" in os.uname()[4]:
    pico_w_os = 1
    print("[+] Running on a Raspi Pico-W")
    from machine import Pin, PWM
    ## NOTE: Pico-W SPECIFIC CONFIGURAITON - CHANGE HERE ##
    RED_PIN = 17
    GREEN_PIN = 22
    BLUE_PIN = 16
    pwm_freq = 1000
    # Configure the pins to be used later in the code
    red_led = Pin(17, mode=Pin.OUT)
    green_led = Pin(22, mode=Pin.OUT)
    blue_led = Pin(16, mode=Pin.OUT)
    pwm__red_led = PWM(red_led)
    pwm__green_led = PWM(green_led)
    pwm__blue_led = PWM(blue_led)
    pwm__red_led.freq(pwm_freq)
    pwm__green_led.freq(pwm_freq)
    pwm__blue_led.freq(pwm_freq)
    # Set the RED/GREEN/BLUE pin variables to the pwm representations of this GPIO interface
    RED_PIN = pwm__red_led
    GREEN_PIN = pwm__green_led
    BLUE_PIN = pwm__blue_led
else:
    raspi_os = 0
    pico_w_os = 0
    print("[-] Not running on ARM OS")
    print("[-] The pigpio library was not loaded")


#########################################################################
# ------------------ Import Libraries Section --------------------------#
#########################################################################

# ----------------
#  Import of libraries for LED driving aspect of code (GPIO pins)
# ----------------
import os
import sys
try:
    import termios
    print("[+] Imported termios library")
except:
    print("[-] Unable to import the termios library")
try:
    import tty
    print("[+] Imported tty library")
except:
    print("[-] Unable to import the tty library")

import time
try:
	from thread import start_new_thread
except:
	from _thread import start_new_thread

# ----------------
#  Import of libraries for audio capturing aspect of code (RCA Line In)
# ----------------
try:
    import alsaaudio, time, audioop
    print("[+] Imported the libraries for audio capturing")
except:
    print("[-] Unable to import the libraries for audio capturing")

# ----------------
#  Import of libraries for converting audio into RGB values
# ----------------
import Psynesthesia.psynesthesia_modified

#########################################################################
# -------------------- LED Control Set-up ------------------------------#
#########################################################################

# ----------------
#  Setting of the defaul values for the RGB + Brightness Settings for driving LEDs
# ----------------
bright = 255
r = 255.0
g = 0.0
b = 0.0

# ----------------
#  Setting of the defaul values for the state variables for driving LEDs
# ----------------
brightChanged = False
abort = False
state = True

# ----------------
#  pi accesses the local Pi's GPIO
# ----------------
if raspi_os != 0:
    pi = pigpio.pi()
else:
    pi = None
    # Use fake class and functions?
    class pi_fake:
        pass
        
        def set_PWM_dutycycle(pin, realBrightness):
            pass

        def stop():
            pass

    pi = pi_fake

# ----------------
#  Set variable to volume based brightness changing
# ----------------
curMaxVal = 0
lstMaxVal = 0

#########################################################################
# ------------------ Audio Capture Set-up ------------------------------#
#########################################################################

if using_microphone_flag:
    print("[*] Using Microphone as Input")
    # --------------------------------------------
    #  Code for setting the properties and attributes for capturing the RCA Line In (e.g. audio)        [ MICROPHONE RECORDING ]
    # --------------------------------------------
    # Open the device in nonblocking capture mode. The last argument could
    # just as well have been zero for blocking mode. Then we could have
    # left out the sleep call in the bottom of the loop
    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,alsaaudio.PCM_NONBLOCK)
    
    # Set attributes: Mono, 8000 Hz, 16 bit little endian samples
    inp.setchannels(2)	# 2 for stereo
    inp.setrate(48000)	# Can't be 8000 since that is for telephony (too slow?)
    inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    
    # The period size controls the internal number of frames per period.
    # The significance of this parameter is documented in the ALSA api.
    # For our purposes, it is ufficient to know that reads from the device
    # will return this many frames. Each frame being 2 bytes long.
    # This means that the reads below will return either 320 bytes of data
    # or 0 bytes of data. The latter is possible because we are in nonblocking
    # mode.
    inp.setperiodsize(160)

#########################################################################
# --------------- Function Definition Section --------------------------#
#########################################################################

# ----------------
#  howToUse function: print out the instructions for how to use the tool
#       Input: None
#       Output: Instruction on how to use the tool
#
#   print ("+ / - = Increase / Decrease brightness")
#   print ("p / r = Pause / Resume")
#   print ("s = Change WAV File")
#   print ("c = Abort Program")
# ----------------
def howToUse():
    # Print out how to use the tool
    print("--------\tUsage Instructions\t--------\n + / -\t=\tIncrease / Decrease brightness\n p / r\t=\tPause / Resume\n s\t=\tChange WAV File\n c\t=\tAbort Program\n")

# ----------------
#  updateColor function: updates the color value for a given color variable
#	Input: color (float varaible), step (int/float variable)
#	Output: returns int/float for the color variable
#
#  Note: If the color value goes above/below the possible max/min value, 
#	then the max/min value is returned
# ----------------
def updateColor(color, step):
	color += step
	
	if color > 255:
		return 255
	if color < 0:
		return 0
		
	return color

# ----------------
#  setLights funciton: sets the PWM cycle for a specific color pin to a certain brightness based
#		on the 'strength'/amount of that given color
#	Input: pin (int variable for GPIO pin), brightness (int/float variable for 
#		'brightness' of a color)
#	Output: None
# ----------------
def setLights(pin, brightness):
    if pico_w_os != 0:
        # Will return a value between 65025 to 0 with scoping to try and provide a range
        realBrightness = int(int(brightness) * (float(65025 / 255.0)))
        # Uses the duty_u16 function to provide PWM control on the Pico-W GPIO
        pin.duty_u16(realBrightness)
    else:
	    # Will return a value between 255 to 0 (int() helps with the (1/255) case = 0)
	    realBrightness = int(int(brightness) * (float(bright) / 255.0))
	    # Starts (non-zero dutycycle) or stops (0) PWM pulses on the GPIO
	    pi.set_PWM_dutycycle(pin, realBrightness)

# ----------------
#  getCh function: obtains and returns a single character (i.e. byte) from stdin
#	Input: None
#	Output: None
# ----------------
def getCh():
	# Set the value of 'fd' to the file descriptor for stdin (i.e. 0)
	fd = sys.stdin.fileno()
	# Command to return a list containing the tty attributes for file descriptor 'fd' as follows:
	#	[iflag, oflag, cflag, lflag, ispeed, ospeed, cc]
	old_settings = termios.tcgetattr(fd)
	# Try statement for setting the properies of the stdin file descriptor
	try:
		# Change the mode of the file descriptor 'fd' to raw
		tty.setraw(fd)
		# Read 1 byte from stdin
		ch = sys.stdin.read(1)
	# Clause that is executed in any event before leaving the try statement
	#  Note: whether an exception (handled or not) has occurred or not
	finally:
		# Set the tty attributes for the file descripttor (fd) from the attributes (old_settings)
		#  Note: The 'when' argument (termios.TCSADRAIN) determines when the attributes change.
		#	-> TCSADRAIN = to change after transmitting all queued output
		termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
	# Return the character (byte) that was read from stdin
	return ch

# ----------------
#  checkKey function:
#	Input: None
#	Output: None
#  Note: Addition of Desire by User to Open a New WAV file
# ----------------
def checkKey():
	# Access the out-of-scope global variables from within this function
	global bright
	global brightChanged
	global state
	global abort
	global curMaxVal	# Grab current maxVal, use that to check against the past value, depending on that go up or down?
	# Maybe just change the bright variable depending on the previous value?

	# While loop for capturing stdin keyboard input
	while True:
		# Grab a single byte character from stdin and set to varaible 'c'
		c = getCh()
		
		# Scenario where '+' is hit and bright less than 255	
		if c == '+' and bright < 255 and not brightChanged:
			brightChanged = True
			time.sleep(0.01)	# Not sure the exact purpose of this sleep. Timing?
			brightChanged = False
			# Set the brightness up by BRIGHT_STEPS (default = 1)
			bright = bright + BRIGHT_STEPS
			print ("Current brightness: %d" % bright)
		
		# Scenario where '-' is hit and bright greater than 0	
		if c == '-' and bright > 0 and not brightChanged:
			brightChanged = True
			time.sleep(0.01)	# Not sure the exact purpose of this sleep. Timing?
			brightChanged = False
			# Set the brightness down by BRIGHT_STEPS (default = 1)	
			bright = bright - BRIGHT_STEPS
			print ("Current brightness: %d" % bright)
		
                # Scenario where 'p' is hit; pauses code running	|   Note: This seems to cause a complete shutdown of the program.... Unable to restart?
		if c == 'p' and state:
			state = False
			print ("Pausing...\n")
			
			time.sleep(0.1)
			# Set all color pin brightness to zero (turn off colors)
			setLights(RED_PIN, 0)
			setLights(GREEN_PIN, 0)
			setLights(BLUE_PIN, 0)
		
		# Scenario where 'r' is hit; resumes code running	
		if c == 'r' and not state:
			state = True
			print ("Resuming...")

		if c == 's' and state:
			if dbg != 0:
				print("User requested new WAV input\n")
			new_wav_file = input("New WAV file: ")
			if dbg != 1:        # ~!~
				print("User provided WAV file\t[\t{0}\t]\n".format(new_wav_file))
			## TODO: Check that the provided file exists, and pass that file to psynethsia code
			time.sleep(0.1)
		
		# Scenario where 'c' is hit; aborts the code running (by setting the abort variable)
		if c == 'c' and not abort:
			abort = True
			break

		# Piece of thread function loop that updates brightness based on the current max value
		# 1) Check current value
		# 2) Compare to total brightness scale (function call to normalize data?)
		# 3) Update last seen value?

#########################################################################
# --------------- Main Functional Code Section -------------------------#
#########################################################################

# ----------------
#  Starting new thread and returning its identifier
#	Note: The thread executes the function (checkKey) with the argument list (args=())
# ----------------
start_new_thread(checkKey, ())	# Printing is weird because I'm seeing the output of the new thread??
# Note: Do I need this piece?
#	-> Helps with starting/stoping the program
#	-> Would allow manual changing of brightness? (Not needed in end product)
#	-> Could use as method for updating brightness of LEDs based on sound

# ----------------
#  Prints out the control information to the stdout (e.g. original terminal that the code is being run from)
# ----------------
#print ("+ / - = Increase / Decrease brightness")
#print ("p / r = Pause / Resume")
#print ("s = Change WAV File")
#print ("c = Abort Program")
howToUse()

# ----------------
#  Set the starting color brightness values for each GPIO pin on the Pi
# ----------------
setLights(RED_PIN, r)
setLights(GREEN_PIN, g)
setLights(BLUE_PIN, b)

# ----------------
#  Check and setting up for an input file light chaing debugging
# ----------------
if not using_microphone_flag:
    input_test_filename="audio-to-rgb.conversion"
    conversion_debugging_input = open(input_test_filename, 'r')
    first_time_completed = 0

# ----------------
#  While loop for constantly changing the RGB mix (e.g. LED display color) constantly over time.
#	Note: This section regulates the PWM on each color GPIO pin
# ----------------
while abort == False:
	# Enter this if statement if the code is not paused (state variable) and the brightness of the LEDs
	#	has not been changed
	if state and not brightChanged:
		# If statement for increasing the PWM for the green GPIO pin
		if r == 255 and b == 0 and g < 255:
			g = updateColor(g, STEPS)
			setLights(GREEN_PIN, g)
		# Elif statement for decreasing the PWM for the red GPIO pin
		elif g == 255 and b == 0 and r > 0:
			r = updateColor(r, -STEPS)
			setLights(RED_PIN, r)
		# Elif statement for increasing the PWM for the blue GPIO pin
		elif r == 0 and g == 255 and b < 255:
			b = updateColor(b, STEPS)
			setLights(BLUE_PIN, b)
		# Elif statement for decreasing the PWM for the green GPIO pin
		elif r == 0 and b == 255 and g > 0:
			g = updateColor(g, -STEPS)
			setLights(GREEN_PIN, g)
		# Elif statement for increasing the PWM for the red GPIO pin
		elif g == 0 and b == 255 and r < 255:
			r = updateColor(r, STEPS)
			setLights(RED_PIN, r)
		# Elif statement for decreasing the PWM for the blue GPIO pin
		elif r == 255 and g == 0 and b > 0:
			b = updateColor(b, -STEPS)
			setLights(BLUE_PIN, b)
	# ??? Place audio capturing code here ??? 
	# ----------------
	#  While loop chunk for reading from the RCA Line In
	# ----------------
	# Read data from device / input
	if using_microphone_flag:
	    if dgb != 0:
	        print("[*] Using the Microphone to Update the Colors (??)\n")
	    # Note: In PCM_NONBLOCK mode, the call will not block, but WILL return (0,'') if no new period has become available since the last call
	    l,data = inp.read()	
	    if dbg != 0:
	    	print ("Value of l:" + str(l) + "\n")
	    	print ("Value of data:" + str(data) + "\n")
	    if l < 0:
	    	continue
	    elif l == 0:
	    	continue
	    elif l:
	    	# Return the maximum of the absolute value of all samples in a fragment.
	    	#print (audioop.max(data, 2))	# Note: Print causes problems with additional tabs added
	    	#print ("Something Cool")
	    	curMaxVal = audioop.max(data, 2)
	# Read input from the debugging file; used as the audio-to-rbg conversion file
	else:
	    if dbg != 0:
	        print("[*] Reading line from input file\n")
	    rgb = conversion_debugging_input.readline()
	    if rgb == "Red\tGreen\tBlue\n":
	        if dbg != 0:
	            print("... Header file.... Ignoring\n")
	    elif len(rgb) == 0:
	        if first_time_completed != 1:
	            if dbg != 1:        # ~!~
	                print("[+] Completed Read through Input File\n")
	            first_time_completed = 1
	        #else:
	            #first_time_completed = 1
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
	        setLights(RED_PIN, r)
	        setLights(GREEN_PIN, g)
	        setLights(BLUE_PIN, b)
	        if dbg != 0:
	            print("... Waiting time {0} seconds before next read\n".format(test_rgb_wait_time_s))
	        if test_sleep_flag != 0:
	            time.sleep(test_rgb_wait_time_s)
	        
	time.sleep(.001)
	
# ----------------
#  Prints out the message "Aborting..." to stdout
# ----------------
print ("Aborting...\n")
if not using_microphone_flag:
    conversion_debugging_input.close()

# ----------------
#  Set all the color GPIO pins to 0 for turning them off at the end of the program
# ----------------
setLights(RED_PIN, 0)
setLights(GREEN_PIN, 0)
setLights(BLUE_PIN, 0)

# ----------------
#  Sleep time (unsure of the reason/purpose for this)
# ----------------
time.sleep(0.5)

# ----------------
#  Stop a Pi connection
# ----------------
pi.stop()
