#!/usr/bin/env python3
import wave,struct,math
# TODO: Add variable input for the .wav file to examine
#waveFile = wave.open("testWavs/piano2.wav", "rb")
waveFile = wave.open("my_record.wav", "rb")
frames = waveFile.getnframes()	# total number of frames/samples
rate = waveFile.getframerate()	# number of frames/samples per second (should be 44100 Hz (44.1 kHz) for CD-quality audio)
length = frames / int(rate)	# length in seconds
channels = waveFile.getnchannels()	# number of channels (should be 2 channels (stereo) for CD-quality audio)
width = waveFile.getsampwidth()	# sample width/bit depth (should be 2 bytes (16 bits) for CD-quality audio)

print("Total Number of Frames:		" + str(frames))
print("Frames Per Second:		" + str(rate))
print("Total Number of Seconds:	" + str(length))
print("Total Number of Channels:	" + str(channels))
print("Bytes Per Frame:		" + str(width))

## Nota Bene: Please note that the sample rate and the bit depth apply to each channel.
##	This means that the total bit rate of a 2-channel, 44.1kHz, 16-bit wav file
##	will be: 2 x 44100 x 16 = 1411200 bits per second = 1411.200 megabits per second
