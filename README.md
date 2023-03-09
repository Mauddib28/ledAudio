# ledAudio

### Set-up Notes and Instructions ###
# Audio LED
1) Set-up capture for RCA Line In and NOT Mic In; done using 'alsamixer' binary
	i) Set 'Input Mux' to 'Line In'
	ii) Set 'Line' to "L	R\nCapture"
2) Run 'arecord' for test
	Ex: arecord -D hw:0,0 -f DAT my_record.wav
[WORKS! Note: Have RCA Line In volume + player (device) volume up all the way]
	-> Note: 'sampleVolume.py' works once the above HW configuration is set

### Section of Odd Notes and Command Examples ###

# Example command for doing 'arecord' of an input connected to the RCA line
arecord -D hw:0,0 -f DAT my_record.wav

# Show wav file properies
play track.wav stat -freq
	-> calculates the input's power spectrum (4096 point DFT) instead of the statictics normally done


### Resources

# How to wire Raspi LEDs
https://dordnung.de/raspberrypi-ledstrip/
