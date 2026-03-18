import os
import numpy
import librosa
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.ndimage import rotate
from PIL import Image


INPUT_FILE = "audio.mp4"
OUTPUT_DIR = "Decoded_Images"

NUM_IMAGES = 10 # Just for testing will adjust later

SCANWIDTH = 3300 # Number of samples to scan for peaks

# make it still readable
THICKNESS = 15 


def load_audio(audio_path):

    #rate, data = wavfile.read(audio_path) using scipy to read the audio file, but it doesn't support mp4 files, so using librosa instead

    data, rate = librosa.load(audio_path, sr=None, mono=True)

    return rate, data


#normalize the audio data to be between 1.0 and -1.0
def normalise(channel):

    # find the maximum absolute value in the channel/ loudest point
    peak = numpy.max(numpy.abs(channel))

    if peak > 0:
        # scale the channel so that the peak is at 1.0 or -1.0
        channel = channel / peak
    
    return channel

# calibration based on the GR because it does not immediately start with images
# it starts with a calibration signal, so we can use that to find the first image and then  
# scan for the next images based on the distance between the peaks of the signal

def find_starting_point(channel, rate):

    # check the first 30sec
    first_30s = channel[:rate*30]

    # find peaks 
    peaks, _ = find_peaks(-first_30s, height = numpy.max(-first_30s) - 0.2)

    if len(peaks) == 0:
        print("No start tone found - decoding from the beginning of the audio")
        return 0
    
    print(f"Start tone found at sample {peaks[-1]:,}")
    return peaks[-1]


print("Loading audio...")
rate, data = load_audio(INPUT_FILE) 
print("Audio loaded. Sample rate:", rate, "Hz")

print("Scanning audio for peaks...")
peaks = find_peaks(data, distance=SCANWIDTH)[0]
print("Peaks found. Found", len(peaks), "peaks")

print("Normalizing audio...")
data = normalise(data)

print("Finding starting point...")
starting_point = find_starting_point(data, rate)
print("Starting point found at sample", starting_point)



# print("Loading audio...")
# rate, data = load_audio(INPUT_FILE)
# print("Audio loaded. Sample rate:", rate, "Hz")

# print("Scanning audio for peaks...")
# peaks = find_peaks(data, distance=SCANWIDTH)[0]
# print("Peaks found. Found", len(peaks), "peaks")


