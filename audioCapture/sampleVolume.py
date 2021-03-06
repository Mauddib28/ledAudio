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
