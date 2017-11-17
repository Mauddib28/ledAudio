#!/usr/bin/python
# -*- coding: utf-8 -*-

#
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


# This script needs running pigpio (http://abyz.co.uk/rpi/pigpio/)


###### CONFIGURE THIS ######

# The Pins. Use Broadcom numbers.
RED_PIN   = 17
GREEN_PIN = 22
BLUE_PIN  = 24

# Number of color changes per step (more is faster, less is slower).
# You also can use 0.X floats.
STEPS     = 50

###### END ######




import os
import sys
import termios
import tty
import pigpio
import time
from thread import start_new_thread

bright = 255
r = 255.0
g = 0.0
b = 0.0

brightChanged = False
abort = False
state = True

pi = pigpio.pi()

def updateColor(color, step):
	color += step
	
	if color > 255:
		return 255
	if color < 0:
		return 0
		
	return color


def setLights(pin, brightness):
	realBrightness = int(int(brightness) * (float(bright) / 255.0))
	pi.set_PWM_dutycycle(pin, realBrightness)


def getCh():
	fd = sys.stdin.fileno()
	old_settings = termios.tcgetattr(fd)
	
	try:
		tty.setraw(fd)
		ch = sys.stdin.read(1)
	finally:
		termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
		
	return ch


def checkKey():
	global bright
	global brightChanged
	global state
	global abort
	
	while True:
		c = getCh()
		
		if c == '+' and bright < 255 and not brightChanged:
			brightChanged = True
			time.sleep(0.01)
			brightChanged = False
			
			bright = bright + 1
			print ("Current brightness: %d" % bright)
			
		if c == '-' and bright > 0 and not brightChanged:
			brightChanged = True
			time.sleep(0.01)
			brightChanged = False
			
			bright = bright - 1
			print ("Current brightness: %d" % bright)
			
		if c == 'p' and state:
			state = False
			print ("Pausing...")
			
			time.sleep(0.1)
			
			setLights(RED_PIN, 0)
			setLights(GREEN_PIN, 0)
			setLights(BLUE_PIN, 0)
			
		if c == 'r' and not state:
			state = True
			print ("Resuming...")
			
		if c == 'c' and not abort:
			abort = True
			break

start_new_thread(checkKey, ())


print ("+ / - = Increase / Decrease brightness")
print ("p / r = Pause / Resume")
print ("c = Abort Program")


setLights(RED_PIN, r)
setLights(GREEN_PIN, g)
setLights(BLUE_PIN, b)


while abort == False:
	if state and not brightChanged:
		if r == 255 and b == 0 and g < 255:
			g = updateColor(g, STEPS)
			setLights(GREEN_PIN, g)
		
		elif g == 255 and b == 0 and r > 0:
			r = updateColor(r, -STEPS)
			setLights(RED_PIN, r)
		
		elif r == 0 and g == 255 and b < 255:
			b = updateColor(b, STEPS)
			setLights(BLUE_PIN, b)
		
		elif r == 0 and b == 255 and g > 0:
			g = updateColor(g, -STEPS)
			setLights(GREEN_PIN, g)
		
		elif g == 0 and b == 255 and r < 255:
			r = updateColor(r, STEPS)
			setLights(RED_PIN, r)
		
		elif r == 255 and g == 0 and b > 0:
			b = updateColor(b, -STEPS)
			setLights(BLUE_PIN, b)
	
print ("Aborting...")

setLights(RED_PIN, 0)
setLights(GREEN_PIN, 0)
setLights(BLUE_PIN, 0)

time.sleep(0.5)

pi.stop()

"""
	Code for the capturing of Audio input
"""

#!/usr/bin/python
## This is an example of a simple sound capture script.
##
## The script opens an ALSA pcm for sound capture. Set
## various attributes of the capture, and reads in a loop,
## Then prints the volume.
##
## To test it out, run it and shour at your microphone (input):

import alsaaudio, time, audioop

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

dbg = 0

while True:
	# Read data from device
	l,data = inp.read()
	if dbg != 0:
		print "Value of l:" + str(l)
		print "Value of data:" + str(data)
	if l < 0:
		continue
	elif l == 0:
		continue
	elif l:
		# Return the maximum of the absolute value of all samples in a fragment.
		print audioop.max(data, 2)
	time.sleep(.001)
