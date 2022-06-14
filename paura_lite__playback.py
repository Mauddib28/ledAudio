# paura_lite:
# An ultra-simple command-line audio recorder with real-time
# spectrogram  visualization

## Notes:
#	Add in a non-blocking mode call using the callback mode
#		-> Done via defininig the callback() method
#		- This should allow for playing of the audio AND visualzer junk

import numpy as np
import pyaudio
import struct
import scipy.fftpack as scp
import termplotlib as tpl
import os
import sys				# For checking script inputs
import wave				# For dealing with WAVE files

### Globals

debugBit = 0
#CHUNK = 1024

### Functions

# Function for playing the audio of a provided WAVE (.wav) file
# Input:    WAVE File
# Output:   Audio played from WAVE file
def audio_analysis__play_wave_file(input_wave_file):
    print("[*] Playing Audio of Provided WAVE File")

    CHUNK = 1024

    wf = wave.open(input_wave_file, 'rb')

    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()

    # open stream (2)
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data
    data = wf.readframes(CHUNK)

    # play stream (3)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)

    # stop stream (4)
    stream.stop_stream()
    stream.close()

    # close PyAudio (5)
    p.terminate()

    print("[+] Completed Playing Audio File")

# Function for creating a waveform spectrum bucket output from a WAVE (.wav) file
#   - Note: This function requires the use of 'Ctrl + C' to exit
# Input:    Microphone / WAVE File
#           I/O Flag    -   Note: Default is Microphone (True)
# Output:   Buckets of Frequencies Present
def audio_analysis__bucketize_wave_file_frequencies(input_file_pointer, io_flag=True):
    print("[*] Creating Bucketized Output from Provided File")
    
    # Check to see which branch of this function will be executed
    if io_flag:
        if debugBit != 0:
            print("\tMoving down the Microphone Input Stage")

        # get window's dimensions
        rows, columns = os.popen('stty size', 'r').read().split()
        
        buff_size = 0.2          # window size in seconds
        wanted_num_of_bins = 40  # number of frequency bins to display

        # initialize soundcard for recording:
        fs = 8000
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=int(fs * buff_size))
	
        while 1:  # for each recorded window (until ctr+c) is pressed
            # get current block and convert to list of short ints,
            block = stream.read(int(fs * buff_size))
            format = "%dh" % (len(block) / 2)
            shorts = struct.unpack(format, block)

            # then normalize and convert to numpy array:
            x = np.double(list(shorts)) / (2**15)
            seg_len = len(x)

            # get total energy of the current window and compute a normalization
            # factor (to be used for visualizing the maximum spectrogram value)
            energy = np.mean(x ** 2)
            max_energy = 0.01  # energy for which the bars are set to max
            max_width_from_energy = int((energy / max_energy) * int(columns)) + 1
            if max_width_from_energy > int(columns) - 10:
                max_width_from_energy = int(columns) - 10

            # get the magnitude of the FFT and the corresponding frequencies
            X = np.abs(scp.fft(x))[0:int(seg_len/2)]
            freqs = (np.arange(0, 1 + 1.0/len(X), 1.0 / len(X)) * fs / 2)

            # ... and resample to a fix number of frequency bins (to visualize)
            wanted_step = (int(freqs.shape[0] / wanted_num_of_bins))
            freqs2 = freqs[0::wanted_step].astype('int')
            X2 = np.mean(X.reshape(-1, wanted_step), axis=1)

            # plot (freqs, fft) as horizontal histogram:
            fig = tpl.figure()
            fig.barh(X2, labels=[str(int(f)) + " Hz" for f in freqs2[0:-1]],
                    show_vals=False, max_width=max_width_from_energy)
            fig.show()
            # add exactly as many new lines as they are needed to
            # fill clear the screen in the next iteration:
            print("\n" * (int(rows) - freqs2.shape[0] - 1))
    else:
        if debugBit != 0:
            print("\tMoving down the Wave Audio File path")
        ## Editted version of the above function
        default_values_flag = 0
        # get window's dimensions
        rows, columns = os.popen('stty size', 'r').read().split()
        ## Nota Bene: This is SPECIFIC TO THE WINDOW RUNNING THE CODE
        #   - The purpose of this is to scale the visualizer to still give a clean and relatively
        #       accurate show of the frequencies and their buckets
        
        # Open the provided wave file pointer
        wf = input_file_pointer

        # Instantiate PyAudio
        p = pyaudio.PyAudio()
        
        '''
        buff_size = 0.2          # window size in seconds
        wanted_num_of_bins = 40  # number of frequency bins to display

        # initialize soundcard for recording:
        fs = 8000  # Frame rate (per second?)
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=int(fs * buff_size))
        '''
        
        # Check which version of variables to implement with
        if default_values_flag != 0:
            CHUNK = 1024                    # Default from the MIT example code
            # open stream (2)	|	Note: Configuration for outputting the audio??
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
	        channels=wf.getnchannels(),
	        rate=wf.getframerate(),
	        output=True)

        else:
            print("\tBucketize Wave File\t-\tVariable Setup")
            fs = wf.getframerate()          # Frame rate (per second?)
            buff_size = 0.2                 # Window size in seconds
            wanted_num_of_bins = 40         # Number of Frequency Bins to Display
            CHUNK = int(fs * buff_size)     # Making CHUNK based on buckets?
            # open stream (2)	|	Note: Configuration for outputting the audio??
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
	        channels=wf.getnchannels(),
	        rate=fs,
	        output=True,
                frames_per_buffer=int(fs * buff_size))

        # Read the data
        data = wf.readframes(CHUNK)
	
#        while 1:  # for each recorded window (until ctr+c) is pressed
        print("Reading Data....", end="")
        while len(data) > 0:    # Read through the provided WAVE file
            # Get the Current Block and Convert to List of Short Ints
            #block = stream.read(int(fs * buff_size))        # Note: Can NOT read from stream... 
            # The above should just be the chunk of data that has been read (since the relative size is the same)
            block = data
            format = "%dh" % (len(block) / 2)
            shorts = struct.unpack(format, block)
            if debugBit != 0:       # ~!~
                print("\t?\t-\tShorts Variable:\t{0}".format(shorts))

            # Then Normalize and Convert to numpy array:
            x = np.double(list(shorts)) / (2**15)
            seg_len = len(x)

            # Get the Total Energy of the Current Window ad Compute a Normalization
            #   Factor (to be used for visualizing the maximum spectrogram value)
            energy = np.mean(x ** 2)
            max_energy = 0.01                   # Energy for which the bars are set to max
            max_width_from_energy = int((energy / max_energy) * int(columns)) + 1
            # Note: The above line is where we need the column varaible set
            if max_width_from_energy > int(columns) - 10:
                max_width_from_energy = int(columns) - 10

            # Get the Magnitude of the FFT and the Corresponding Frequencies
            X = np.abs(scp.fft(x))[0:int(seg_len/2)]
            freqs = (np.arange(0, 1 + 1.0/len(X), 1.0 / len(X)) * fs / 2)

            # Debug Check
            if debugBit != 1:       # ~!~
                print("\t?\t-\tfreqs:\t{0}".format(freqs))
                print("\t?\t-\tfreqs.shape[0]:\t{0}\n\t\t\twanted_num_of_bins:\t{1}".format(freqs.shape[0], wanted_num_of_bins))

            # ... and Re-Sample to a fixed number of frequency bins (to visualize)
            wanted_step = (int(freqs.shape[0] / wanted_num_of_bins))
            freqs2 = freqs[0::wanted_step].astype('int')

            # Debug Check
            if debugBit != 1:       # ~!~
                print("\t?\t-\twanted_step:\t{0}\n\t\t\tfreqs2:\t{1}".format(wanted_step, freqs2))

            '''
            X2 = np.mean(X.reshape(-1, wanted_step), axis=1)

            # Plot (freqs, fft) as a Horizontal Histogram:
            fig = tpl.figure()
            fig.barh(X2, labels=[str(int(f)) + " Hz" for f in freqs2[0:-1]],
                    show_vals=False, max_width=max_width_from_energy)
            fig.show()
            '''

            # Before restarting the loop, grab the next piece of data
            print(".", end="")
            data = wf.readframes(CHUNK)
            '''
            freqs = (np.arange(0, 1 + 1.0/len(X), 1.0 / len(X)) * fs / 2)

            # ... and resample to a fix number of frequency bins (to visualize)
            wanted_step = (int(freqs.shape[0] / wanted_num_of_bins))
            freqs2 = freqs[0::wanted_step].astype('int')
            X2 = np.mean(X.reshape(-1, wanted_step), axis=1)

            # plot (freqs, fft) as horizontal histogram:
            fig = tpl.figure()
            fig.barh(X2, labels=[str(int(f)) + " Hz" for f in freqs2[0:-1]],
                    show_vals=False, max_width=max_width_from_energy)
            fig.show()
            # add exactly as many new lines as they are needed to
            # fill clear the screen in the next iteration:
            print("\n" * (int(rows) - freqs2.shape[0] - 1))
            '''
        print("")

    print("[+] Completed Frequency Bucketization")

# Function for checking the input provided to the script
# Input:    None
# Output:   Works or Does not
def check_function__script_start():
    print("[*] Beginning Script Checks")
    # Function for looking for a Python argument to this script
    if len(sys.argv) < 2:
        print("Attempts to read, play, and visualize some audio data.\n\nUsage: {0} filename.wav".format(sys.argv[0]))
        sys.exit(-1)
    else:
        input_audio_file = sys.argv[1]
        print("[+] Setting the input audio file to:\t{0}".format(input_audio_file))

    print("[+] Compelted Script Checks..... Continuing Code")
    return input_audio_file

### Main Code

input_audio_file = check_function__script_start()

# Setting the input WAV file for openning
wf = wave.open(input_audio_file, 'rb')

#audio_analysis__play_wave_file(wf)
#audio_analysis__bucketize_wave_file_frequencies(wf, io_flag=True)
io_flag = False
audio_analysis__bucketize_wave_file_frequencies(wf, io_flag)

'''
OLD CODE FOR BUCKET-IZING THE MICROPHONE

# get window's dimensions
rows, columns = os.popen('stty size', 'r').read().split()

buff_size = 0.2          # window size in seconds
wanted_num_of_bins = 40  # number of frequency bins to display

# variable for frequency analysis
#	- Note: Comes out of the PyAudio example + Github search / find
fs = wf.getsampwidth()/buff_size

## Note: The below does the basic of reading in an audio file and playing it

# initialize the PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2)	|	Note: Configuration for outputting the audio??
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
	channels=wf.getnchannels(),
	rate=wf.getframerate(),
	output=True)

# read data
data = wf.readframes(CHUNK)

## play stream (3)
while len(data) > 0:
	# Playing audio
	stream.write(data)
	data = wf.readframes(CHUNK)

	## Write out interesting frequency information
	# get current block and convert to list of short ints,
	block = stream.read(int(fs * buff_size))
	format = "%dh" % (len(block) / 2)
	shorts = struct.unpack(format, block)

	# then normalize and convert to numpy array:
	x = np.double(list(shorts)) / (2**15)
	seg_len = len(x)

	# get total energy of the current window and compute a normalization
	# factor (to be used for visualizing the maximum spectrogram value)
	energy = np.mean(x ** 2)
	max_energy = 0.01  # energy for which the bars are set to max
	max_width_from_energy = int((energy / max_energy) * int(columns)) + 1
	if max_width_from_energy > int(columns) - 10:
	    max_width_from_energy = int(columns) - 10

	# get the magnitude of the FFT and the corresponding frequencies
	X = np.abs(scp.fft(x))[0:int(seg_len/2)]
	freqs = (np.arange(0, 1 + 1.0/len(X), 1.0 / len(X)) * fs / 2)

	# ... and resample to a fix number of frequency bins (to visualize)
	wanted_step = (int(freqs.shape[0] / wanted_num_of_bins))
	freqs2 = freqs[0::wanted_step].astype('int')
	X2 = np.mean(X.reshape(-1, wanted_step), axis=1)

	# plot (freqs, fft) as horizontal histogram:
	fig = tpl.figure()
	fig.barh(X2, labels=[str(int(f)) + " Hz" for f in freqs2[0:-1]],
	         show_vals=False, max_width=max_width_from_energy)
	fig.show()
	# add exactly as many new lines as they are needed to
	# fill clear the screen in the next iteration:
	print("\n" * (int(rows) - freqs2.shape[0] - 1))


# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()

'''
