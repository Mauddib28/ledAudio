###
# Editted version of the original Psynethesia by rho-bit
#   - Changed output to RGB tuple 
#   - Removed pygame output to minimize resource usage
##

### Imports
import pyaudio              # Import for pyaudio?
#import pygame               # Import for pygame window
import wave                 # Import for WAVE file I/O
import numpy as np          # Import for FFT stuff
from wavtorgb import *      # Import from rho-bit written code
from math import *          # Import from math libraries

### Globals
print("[*] Setting up the Global Variables")
debugBit = 0
enableAudioBit = 0
# Note: The output format here is in the form RGB (Red, Green, Blue)
testOutput_conversionFile='./audio-to-rgb.conversion'

#Request for the file name of the WAV file.
print("[*] Requesting the input WAV file")
raw = input("WAV file name?: ")
#Starts Pygame and opens the screen 
#pygame.init()
#screen = pygame.display.set_mode((800, 800))

chunk = 2048
# open the WAV file
wf = wave.open(raw, 'rb')
swidth = wf.getsampwidth()
RATE = wf.getframerate()
# use a Blackman window
window = np.blackman(chunk)
# open the stream
p = pyaudio.PyAudio()
#background = pygame.Surface(screen.get_size())
#background = background.convert()

thefreq = 1.0
#global thefreq
## Check if an audio output is expected or not
#if enableAudioBit != 0:
    # Playing the Audio will Occur
# Openning the stream
print("[*] Openning the PyAudio stream")
stream = p.open(format =
                p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = RATE,
                output = True)
#else:
    # Audio will NOT be Playe
#    stream = p.open(format =
#                p.get_format_from_width(wf.getsampwidth()),
#                channels = wf.getnchannels(),
#                rate = RATE,
#                input = True)

# Creating the conversion output file
#   - TODO: Have this be a check for existing file instead of flat overwrite
print("[*] Creating the Audio-to-RGB Output File")
conversion_output = open(testOutput_conversionFile, 'w')
fast_overwrite = "Red\tGreen\tBlue\n"
conversion_output.writelines(fast_overwrite)
conversion_output.close()

print("[*] Reading the first chunk of data")
# read the incoming data
data = wf.readframes(chunk)
print("[*] Starting the while loop for the Audio Stream")
if debugBit != 1:   # ~!~
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
        print("the previous freq is "+str(thefreq))
        while thefreq < 350 and thefreq > 15:
            #global thefreq
            thefreq = thefreq*2
            print("the new freq is "+str(thefreq)) 
        while thefreq > 700:
            #global thefreq
            thefreq = thefreq/2
            print("the new freq is"+str(thefreq)) 
        c = 3*10**8
        THz = thefreq*2**40
        pre = float(c)/float(THz)
        nm = int(pre*10**(-floor(log10(pre)))*100)	
        print("Your nm total: "+str(nm))
        rgb = wavelen2rgb(nm, MaxIntensity=255)
        print("the colors for this nm are: "+str(rgb))
        #Fills the background with the appropriate colot, does this so fast, it creates a "fading effect" in between colors
        #background.fill((rgb[0],rgb[1],rgb[2]))
        # Debug output for checking result of 'rgb' variable (return from wavelen2rgb)
        if debugBit != 1:       # ~!~
            print("[+] Colors Generated:\t{0}\n\tColor 0 (Red):\t{1}\n\tColor 1 (Green):\t{2}\n\tColor 2 (Blue):\t{3}".format(rgb, rgb[0], rgb[1], rgb[2]))
            print("... Writing Conversion to Output File [ {0} ]".format(testOutput_conversionFile))
            conversion_output = open(testOutput_conversionFile, 'a')
            conversion_line = "{0}\t{1}\t{2}\n".format(rgb[0], rgb[1], rgb[2])
            conversion_output.writelines(conversion_line)
            conversion_output.close()
        #"blits" (renders) the color to the background
        #screen.blit(background, (0, 0))
        #and finally displays the background
        #pygame.display.flip()
	
    print("\tReading Next chunk of Data")	
    # read some more data
    data = wf.readframes(chunk)

# If there is data write it out the stream
print("[*] Attempting to write last piece of data")
if data:
    stream.write(data)

print("[*] Closing Stream and PyAudio instance")
stream.close()
p.terminate()

print("[+] Terminating Psynetheia Modified Script")
