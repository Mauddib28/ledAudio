###
# Editted version of the original Psynethesia by rho-bit
#   - Changed output to RGB tuple 
#   - Removed pygame output to minimize resource usage
##

## TODO: Fix up this file into the same amount of structure as the music_react script
## TODO: Fix so that psynethesia works with non-mono track audio

#########################################################################
# ------------------ Import Libraries Section --------------------------#
#########################################################################

# ----------------
#  Import of libraries for intaking a .WAV file and converting frequencies to colors
# ----------------
import pyaudio              # Import for pyaudio?
# Testing for pygame library existing on the target system
try:
    import pygame               # Import for pygame window
    pygame_testing = 0
except:
    print("[-] Did not load the pygame library")
    pygame_testing = 1
import wave                 # Import for WAVE file I/O
import numpy as np          # Import for FFT stuff
# Attempt to import the wavtorgb functionality based on where this script is getting called
try:
    from wavtorgb import *      # Import from rho-bit written code
    print("[*] Importing wavtorgb locally")
except:
    from Psynesthesia.wavtorgb import *     # Import from rho-bit written code
    print("[*] Importing wavtorgb from Psynesthesia")
from math import *          # Import from math libraries

# Imports for Arguments to Python Script
import getopt                   # Helps with argument passing
import sys                      # Allows for passing arguments

# Imports for Pipe Usage
import os                       # Helps with creating and writing to a named pipe file

#########################################################################
# ---------------- Globals Definition Section --------------------------#
#########################################################################
print("[*] Setting up the Global Variables")
debugBit = 0
enableAudioBit = 0
# Note: The output format here is in the form RGB (Red, Green, Blue)
#test_conversion_file='./audio-to-rgb.conversion'
test_conversion_file='/tmp/audio-to-rgb.conversion'
default_conversion_pipe='/tmp/audio-to-rgb.pipe'

#########################################################################
# --------------- Function Definition Section --------------------------#
#########################################################################

## Function for Creating the test Audio-to-RGB Output File
def create__audio_to_rgb_conversion_file(testOutput_conversionFile):
    # Creating the conversion output file
    #   - TODO: Have this be a check for existing file instead of flat overwrite
    print("[*] Creating the Audio-to-RGB Output File")
    conversion_output = open(testOutput_conversionFile, 'w')
    fast_overwrite = "Red\tGreen\tBlue\n"
    conversion_output.writelines(fast_overwrite)
    conversion_output.close()

## Function for requesting the input WAV file
def request__input_wav_file():
    print("[*] Requesting the input WAV file")
    # Request the file name of the WAV file
    raw = input("WAV file name?: ")
    #Starts Pygame and opens the screen 
    if pygame_testing != 0:
        pygame.init()
        screen = pygame.display.set_mode((800, 800))
    return raw

## Function for Openning the PyAudio stream for use in this script
def open__pyaudio_stream(p, wf, RATE):
    print("[*] Openning the PyAudio stream")
    # Opening the stream
    stream = p.open(format =
                p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = RATE,
                output = True)
    return stream
#else:
    # Audio will NOT be Playe
#    stream = p.open(format =
#                p.get_format_from_width(wf.getsampwidth()),
#                channels = wf.getnchannels(),
#                rate = RATE,
#                input = True)

## Function for Performing Audio-to-RGB Conversion to an Output File    -   NOTE: NO AUDIO OUTPUT                   [   PURPOSE IS FOR TESTING  ]
def create_and_exploit__audio_to_rgb(testOutput_conversionFile, wf, RATE, chunk, swidth, window, thefreq, stream, background=None):
    print("[*] Reading the first chunk of data")
    # read the incoming data
    data = wf.readframes(chunk)
    print("[*] Starting the while loop for the Audio Stream")
    if debugBit != 0:   # ~!~
        print("[?] Test Pre-While Loop")
        print("\tChunk:\t{0}\n\tData:\t{1}\n\tSWidth:\t{2}\n\tLen(Data):\t{3}\n\tChunk*SWidth:\t{4}".format(chunk, data, swidth, len(data), chunk*swidth))
    # play stream and find the frequency of each chunk
    while len(data) == chunk*swidth:
        if debugBit != 0:
            print("\tWrite data out to the Stream")
        # write data out to the audio stream
        stream.write(data)
        # unpack the data and times by the hamming window
        indata = np.array(wave.struct.unpack("%dh"%(len(data)/swidth),\
                                             data))*window
        # Take the fft and square each value
        fftData=abs(np.fft.rfft(indata))**2
        # find the maximum
        which = fftData[1:].argmax() + 1
        # use quadratic interpolation around the max
        if which != len(fftData)-1:
            y0,y1,y2 = np.log(fftData[which-1:which+2:])
            x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
            # find the frequency and output it
            thefreq = (which+x1)*RATE/chunk
            thefreq = which*RATE/chunk
            if debugBit != 0:
                print("the previous freq is "+str(thefreq))
            while thefreq < 350 and thefreq > 15:
                #global thefreq
                thefreq = thefreq*2
                if debugBit != 0:
                    print("the new freq is "+str(thefreq)) 
            while thefreq > 700:
                #global thefreq
                thefreq = thefreq/2
                if debugBit != 0:
                    print("the new freq is"+str(thefreq)) 
            c = 3*10**8
            THz = thefreq*2**40
            pre = float(c)/float(THz)
            nm = int(pre*10**(-floor(log10(pre)))*100)	
            if debugBit != 0:
                print("Your nm total: "+str(nm))
            rgb = wavelen2rgb(nm, MaxIntensity=255)
            if debugBit != 0:
                print("the colors for this nm are: "+str(rgb))
            #Fills the background with the appropriate colot, does this so fast, it creates a "fading effect" in between colors
            if pygame_testing != 0:
                background.fill((rgb[0],rgb[1],rgb[2]))
            # Debug output for checking result of 'rgb' variable (return from wavelen2rgb)
            if debugBit != 0:       # ~!~
                print("[+] Colors Generated:\t{0}\n\tColor 0 (Red):\t\t{1}\n\tColor 1 (Green):\t{2}\n\tColor 2 (Blue):\t\t{3}".format(rgb, rgb[0], rgb[1], rgb[2]))
                print("... Writing Conversion to Output File [ {0} ]".format(testOutput_conversionFile))
            conversion_output = open(testOutput_conversionFile, 'a')
            conversion_line = "{0}\t{1}\t{2}\n".format(rgb[0], rgb[1], rgb[2])
            conversion_output.writelines(conversion_line)
            conversion_output.close()
            if pygame_testing != 0:
                #"blits" (renders) the color to the background
                screen.blit(background, (0, 0))
                #and finally displays the background
                pygame.display.flip()
    	
        if debugBit != 0:
            print("\tReading Next chunk of Data")	
        # read some more data
        data = wf.readframes(chunk)

    # If there is data write it out the stream
    print("[*] Attempting to write last piece of data")
    if data:
        stream.write(data)

## Function for running the Audio-to-RGB Psynethsia code through a given test_conversion_file
def audio_to_rgb__mode__conversion_test_file(test_conversion_file, raw=None):
    if raw == None:
        print("[*] Running Audio-to-RGB\t-\tMode:\tConversion Test File")
        # Request for the file name of the WAV file.
        raw = request__input_wav_file()
    else:
        print("[*] Running Audio-to-RGB\t-\tMode:\tKnown Input/Output Conversion")
        print("\tInfile:\t{0}".format(raw))
    
    # Setup variables and such for the function of this script
    chunk = 2048
    # open the WAV file
    wf = wave.open(raw, 'rb')
    swidth = wf.getsampwidth()
    RATE = wf.getframerate()
    # use a Blackman window
    window = np.blackman(chunk)
    # open the stream
    p = pyaudio.PyAudio()
    if pygame_testing != 0:
        background = pygame.Surface(screen.get_size())
        background = background.convert()
    
    thefreq = 1.0
    #global thefreq
    ## Check if an audio output is expected or not
    #if enableAudioBit != 0:
        # Playing the Audio will Occur
    
    # Openning the stream
    stream = open__pyaudio_stream(p, wf, RATE)

    # Pipe testing flag
    pipeBit = 1
   
    if pipeBit != 1:
        # Create the test audio-to-rgb conversion file
        create__audio_to_rgb_conversion_file(test_conversion_file)
    else:
        try:
            # Create the test audio-to-rgb conversion pipe
            os.mkfifo(test_conversion_file, mode=0o666)
        except FileExistsError:
            print("Named pipe already exists at:\t{0}".format(test_conversion_file))
        except OSError as e:
            print("OS Error:\t{0}".format(e))
    
    # Perform Audio-to-RGB Conversion - Specifically writing the result to an output file                               [   PURPOSE IS FOR TESTING  ]
    if pygame_testing != 0:
        create_and_exploit__audio_to_rgb(test_conversion_file, wf, RATE, chunk, swidth, window, thefreq, stream, background)
    else:
        create_and_exploit__audio_to_rgb(test_conversion_file, wf, RATE, chunk, swidth, window, thefreq, stream)
    
    ## TODO: Create an Audio-to-RGB 

    if pipeBit != 0:
        # Close the pipe
        os.remove(test_conversion_file)
    
    print("[*] Closing Stream and PyAudio instance")
    stream.close()
    p.terminate()
    
    print("[+] Terminating Psynetheia Modified Script")

## Function for Performing Audio-to-RGB Conversion to a PIPE File
def create_and_exploit__audio_to_rgb__output_pipe(output_conversion_pipe, wf, RATE, chunk, swidth, window, thefreq, stream, background=None):
    print("[*] Reading the first chunk of data")
    # read the incoming data
    data = wf.readframes(chunk)
    print("[*] Starting the while loop for the Audio Stream")
    if debugBit != 0:   # ~!~
        print("[?] Test Pre-While Loop")
        print("\tChunk:\t{0}\n\tData:\t{1}\n\tSWidth:\t{2}\n\tLen(Data):\t{3}\n\tChunk*SWidth:\t{4}".format(chunk, data, swidth, len(data), chunk*swidth))
    # play stream and find the frequency of each chunk
    while len(data) == chunk*swidth:
        print("\tWrite data out to the Stream")
        # write data out to the audio stream
        stream.write(data)
        # unpack the data and times by the hamming window
        indata = np.array(wave.struct.unpack("%dh"%(len(data)/swidth),\
                                             data))*window
        # Take the fft and square each value
        fftData=abs(np.fft.rfft(indata))**2
        # find the maximum
        which = fftData[1:].argmax() + 1
        # use quadratic interpolation around the max
        if which != len(fftData)-1:
            y0,y1,y2 = np.log(fftData[which-1:which+2:])
            x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
            # find the frequency and output it
            thefreq = (which+x1)*RATE/chunk
            thefreq = which*RATE/chunk
            if debugBit != 0:
                print("the previous freq is "+str(thefreq))
            while thefreq < 350 and thefreq > 15:
                #global thefreq
                thefreq = thefreq*2
                if debugBit != 0:
                    print("the new freq is "+str(thefreq)) 
            while thefreq > 700:
                #global thefreq
                thefreq = thefreq/2
                if debugBit != 0:
                    print("the new freq is"+str(thefreq)) 
            c = 3*10**8
            THz = thefreq*2**40
            pre = float(c)/float(THz)
            nm = int(pre*10**(-floor(log10(pre)))*100)	
            if debugBit != 0:
                print("Your nm total: "+str(nm))
            rgb = wavelen2rgb(nm, MaxIntensity=255)
            if debugBit != 0:
                print("the colors for this nm are: "+str(rgb))
            #Fills the background with the appropriate colot, does this so fast, it creates a "fading effect" in between colors
            if pygame_testing != 0:
                background.fill((rgb[0],rgb[1],rgb[2]))
            # Debug output for checking result of 'rgb' variable (return from wavelen2rgb)
            if debugBit != 0:       # ~!~
                print("[+] Colors Generated:\t{0}\n\tColor 0 (Red):\t\t{1}\n\tColor 1 (Green):\t{2}\n\tColor 2 (Blue):\t\t{3}".format(rgb, rgb[0], rgb[1], rgb[2]))
                print("... Writing Conversion to Output File [ {0} ]".format(output_conversion_pipe))
            # Perform RBG Conversion to the output pipe
            conversion_output = open(output_conversion_pipe, 'a')
            conversion_line = "{0}\t{1}\t{2}\n".format(rgb[0], rgb[1], rgb[2])
            conversion_output.writelines(conversion_line)
            conversion_output.close()
            if pygame_testing != 0:
                #"blits" (renders) the color to the background
                screen.blit(background, (0, 0))
                #and finally displays the background
                pygame.display.flip()
    	
        print("\tReading Next chunk of Data")	
        # read some more data
        data = wf.readframes(chunk)

    # If there is data write it out the stream
    print("[*] Attempting to write last piece of data")
    if data:
        stream.write(data)

## Function for running the Audio-to-RGB Psynethsia code through a given test_conversion_pipe
def audio_to_rgb__mode__conversion_pipe_output(test_conversion_pipe):
    print("[*] Running Audio-to-RGB\t-\tMode:\tConversion Pipe Output")
    # Request for the file name of the WAV file.
    raw = request__input_wav_file()
    
    # Setup variables and such for the function of this script
    chunk = 2048
    # open the WAV file
    wf = wave.open(raw, 'rb')
    swidth = wf.getsampwidth()
    RATE = wf.getframerate()
    # use a Blackman window
    window = np.blackman(chunk)
    # open the stream
    p = pyaudio.PyAudio()
    if pygame_testing != 0:
        background = pygame.Surface(screen.get_size())
        background = background.convert()
    
    thefreq = 1.0
    #global thefreq
    ## Check if an audio output is expected or not
    #if enableAudioBit != 0:
        # Playing the Audio will Occur
    
    # Openning the stream
    stream = open__pyaudio_stream(p, wf, RATE)
    
    # Create the test audio-to-rgb conversion file
    create__audio_to_rgb_conversion_file(test_conversion_pipe)
    
    # Perform Audio-to-RGB Conversion - Specifically writing the result to an output file                               [   PURPOSE IS FOR TESTING  ]
    if pygame_testing != 0:
        create_and_exploit__audio_to_rgb(test_conversion_pipe, wf, RATE, chunk, swidth, window, thefreq, stream, background)
    else:
        create_and_exploit__audio_to_rgb(test_conversion_pipe, wf, RATE, chunk, swidth, window, thefreq, stream)
    
    ## TODO: Create an Audio-to-RGB 
    
    print("[*] Closing Stream and PyAudio instance")
    stream.close()
    p.terminate()
    
    print("[+] Terminating Psynetheia Modified Script")

#########################################################################
# --------------- Main Functional Code Section -------------------------#
#########################################################################

# Main Function
#   -> Where calls are made for running the Audio-to-RGB in various modes
def main(argv):
    # Variables
    input_file, output_file = None, None
    # Check for inputs
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["in-file", "out-file"])
        # Parse Arguments
        for opt, arg in opts:
            # Help Menu
            if opt == '-h':
                print('./psynesthesia_modified.py -i <in-file.wav> -o <out-file.conversion>')
                sys.exit()
            # Check for I/O mode or basic testing mode
            else:
                if opt in ("-i", "--in-file"):
                    input_file = arg
                    print("[+] Obtained input file:\t{0}".format(input_file))
                if opt in ("-o", "--out-file"):
                    output_file = arg
                    print("[+] Obtained output file:\t{0}".format(output_file))
    except getopt.GetoptError:
        print('./psynesthesia_modified.py -h for help')
        print("Without arguments, one can run in testing mode with the following bash commands\ntestFile = \"/home/duncan/Documents/LifeShit/Projects/RaspiFun/ledAudio/testWavs/i_ran_so_far_away-flock_of_seagulls.wav\"\necho \"$testfile\" | python3 ./psynesthesia_modified.py")
        sys.exit()      # Exit if weird flags presented
    ## Sanity check operating mode
    # Output to File vs Stream/Pipe

    ## File Output
    # Test File Mode
    if (input_file == None) or (output_file == None):
        print("[*] Operating in basic test mode")
        # Main Code Function
        debugBit = 1
        print("[*] Beginning Main Function")
        # Run test of the Audio-to-RGB Conversion File functionality
        audio_to_rgb__mode__conversion_test_file(test_conversion_file)
        print("[+] Completed Main Function")
    # Input/Output File Mode
    else:
        print("[*] Operating with knnown input and output mode")
        # Main Code Function
        debugBit = 0
        # Run Audio-to-RGB Conversion with known input and output
        audio_to_rgb__mode__conversion_test_file(output_file, input_file)
        print("[+] Completed RGB Conversion")

    ## Stream/Pipe Output - Regular vs BLE (Note: Use select.select to read form file); MAybe not needed??? Already built into the function?

## Actual Main Code That Gets Run
if __name__ == "__main__":
    if debugBit != 0:
        print("[*] Script Being Run - Not Being Imported (???)")
    main(sys.argv[1:])
