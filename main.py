import os
import numpy
import librosa
from scipy.signal import find_peaks
from PIL import Image


INPUT_FILE = "audio.wav"
OUTPUT_DIR = "Decoded_Images"

NUM_IMAGES = 15 # set to None to decode all

#CANWIDTH = 3300 # Number of samples to scan for peaks

# make it still readable
# 1 keeps the native ~4:3 aspect ratio of the voyager images
THICKNESS = 1


def load_audio(audio_path):

    #rate, data = wavfile.read(audio_path) using scipy to read the audio file, but it doesn't support mp4 files, so using librosa instead

    data, rate = librosa.load(audio_path, sr=None, mono=False)

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
    peaks, _ = find_peaks(-first_30s, height = numpy.max(-first_30s) - 0.2) # _ is just a throwaway var becuase find_peaks gives us a dict of properties we dont need, we just want the peaks

    if len(peaks) == 0:
        print("No start tone found - decoding from the beginning of the audio")
        return 0
    
    print(f"Start tone found at sample {peaks[-1]:,}")
    return peaks[-1]


# use the loud beeps between images to find the boundaries

def find_image_bounderies(channel, rate):
    # the boundary pulses must be at least 5 seconds apart (distance = rate * 5).
    # i originally thought this was 1/5 of a second, but images are actually spaced 
    # several seconds apart on the record. this prevents us from picking up internal 
    # image sync pulses as false boundaries.

    peaks, _ = find_peaks(channel, height = numpy.max(channel) - 0.2, distance = rate * 5) 

    print(f"Found {len(peaks)} potential image boundary peaks")
    return peaks


# Extract the images
def decode_image(img_signal, rate, img_num):

    min_distance = int(rate * 0.008) #min distance btw scanlines is roughly 8ms
    
    # look only in the first tenth of the signal to find the boundary beep and skip it.
    # if we don't chop off the loud boundary beep, the local normalisation below will fail
    # because the beep is way louder than the image data.
    skip_samples = int(rate * 0.1)
    if len(img_signal) > skip_samples:
        img_signal = img_signal[skip_samples:]

    # some local normalisation
    img_signal = normalise(img_signal)
    
    raw_peaks, _ = find_peaks(img_signal, height = 0.2, distance = min_distance)

    # if we find less than 4 scan lines, something went wrong - skip the image....
    if len(raw_peaks) < 4:
        print(f"Image {img_num} too few scan lines({len(raw_peaks)}) - skipping")
        return None
    #Jitterrrrrrrrrrrrrrrr
    # the sync pulses alternate in width. if we align the image to the peak, the changing
    # width shifts every other line left/right causing an echo/jitter effect.
    # so we search slightly past the peak to find the bottom of the pulse (trough). 
    # the falling edge is perfectly stable.
    troughs =[]
    search_window = int(rate * 0.002) 
    for peak in raw_peaks:
        if peak + search_window < len(img_signal):
            trough_offset = numpy.argmin(img_signal[peak : peak + search_window])
            troughs.append(peak + trough_offset)
            
    troughs = numpy.array(troughs)

    diffs = numpy.diff(troughs)
    median_diff = numpy.median(diffs)

    # filter out missed pulses to get a perfect average
    valid_diffs = diffs[numpy.abs(diffs - median_diff) < (rate * 0.001)]
    
    if len(valid_diffs) == 0:
        true_samples_per_line = int(rate * 0.008333)
    else:
        true_samples_per_line = int(numpy.round(numpy.mean(valid_diffs)))

    #make the image grid
    rows =[]
    sync_offset = 30 
    drift_allowance = int(rate * 0.0002) #  # a tiny window to track natural speed drift in the audio

    # skip the first 2 and the last 2 peaks, because they are usually partial or noisy
    current_trough = troughs[2] 
    
    # voyager images always have exactly 512 lines. 
    # instead of just trusting find_peaks (which broke most of the way through some images) 
    # we predict exactly where the next line should be. if bright image data ruins the sync pulse, 
    # we just coast blindly forward on our prediction until the real sync pulses come back.
    # don't mind the comments I like to leave notes to myself if something gets complicated, because I will most likely forget it after 48hrs
    for _ in range(512):
        start_idx = current_trough - sync_offset
        end_idx = start_idx + true_samples_per_line
        
        if start_idx < 0 or end_idx > len(img_signal):
            break
            
        line = img_signal[start_idx : end_idx] # extract the line
        for _ in range(THICKNESS):
            rows.append(line)
        
        # predict where the next sync pulse should be based on our dynamic line width calculation
        expected_next = current_trough + true_samples_per_line

        # look inside a tiny micro-window around the prediction 
        window_start = expected_next - drift_allowance
        window_end = expected_next + drift_allowance
        
        if window_start >= 0 and window_end < len(img_signal):
             # find the lowest point in this tiny window
            local_offset = numpy.argmin(img_signal[window_start:window_end])
            actual_trough = window_start + local_offset

            # sanity check- is this local minimum actually a deep negative sync pulse?
            # if the image data is solid bright white, the local minimum won't be a deep negative.
            if img_signal[actual_trough] < -0.05:
                current_trough = actual_trough # lock onto the real sync pulse 
            else:
                current_trough = expected_next 
        else:
            current_trough = expected_next

    # if no valid rows were found then skip the image
    if not rows:
        print(f"Image {img_num} no valid scan lines found - skipping")
        return None
    
    canvas = numpy.array(rows)
    decoded_lines = len(rows) // THICKNESS
    print(f"Image {img_num}: {decoded_lines} scanlines decoded")
    # Voyager scans bottom-to-top AND right-to-left. 
    # Applying BOTH flips fixes the upside-down and the mirroring. after trying
    #canvas = numpy.flipud(canvas)
    #canvas = numpy.fliplr(canvas)

    # I am leaving the above here for reminders... haha

    # i originally thought we needed to flip the array all over the place to fix the rotation, 
    # but the GR just draws images in vertical columns (top to bottom, left to right). 
    # transposing the 2D array (.T) perfectly maps the audio lines to vertical image columns.
    canvas = canvas.T
    

    # some contrasts and brightness adjustments to make the image more visible
    # clip the darkest 1% and brightest 1% of pixels to remove extreme noise spikes
    lo = numpy.percentile(canvas, 1)
    hi = numpy.percentile(canvas, 99)
    canvas = numpy.clip(canvas, lo, hi)

    # scale everything between 0.0 and 1.0
    canvas = (canvas - lo) / (hi - lo + 1e-9)
    
    # we subtract the pixels from 1.0 to invert the colors
    canvas = 1.0 - canvas
    
    canvas = (canvas ** 0.8) * 255

    return canvas.astype(numpy.uint8)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rate, data = load_audio(INPUT_FILE)

    decoded_total = 0

    # the Golden Record audio is a stereo track... left and right channel...if the audio is mono,
    # we can just use the single channel, but if it's stereo, we need to split it into two channels and decode them separately
    if data.ndim == 2:
        channels = [("left", data[0]), ("right", data[1])]
    else:
        channels = [("mono", data)]

    for ch_name, raw_channel in channels:

        # how many more images we need to decode from this channel
        want = None if NUM_IMAGES is None else max(0, NUM_IMAGES - decoded_total)

        # stop processing if we have decoded enough images
        if want == 0:
            break
        print(f"processing channel {ch_name}...")

        # scale the audio so that its loudest point is 1.0
        channel = normalise(raw_channel)

        # find the opening calibration tone and skip past it
        start = find_starting_point(channel, rate)
        channel = channel[start:] # chop off everything before the images

        # find the loud boundary pulses that seperate each image
        boundaries = find_image_bounderies(channel, rate)

        if len(boundaries) == 0:
            print(f"No image boundary peaks found {ch_name} - skipping\n")
            continue

        # endpoint list for each image segment by shifting the boundary array by 1
        ends = numpy.append(boundaries[1:], len(channel)) 

        # how manyimgaes we have decoded from this channel
        limit = want if want is not None else len(boundaries)

        for start_idx, endidx in zip(boundaries[:limit], ends[:limit]):

            img_signal = channel[start_idx:endidx]

            img = decode_image(img_signal, rate, decoded_total + 1)

            if img is None:
                continue
            
            out_path = os.path.join(OUTPUT_DIR, f"golden_record_{decoded_total + 1:03d}.png")

             # convert the raw array into a grayscale image and save it
            Image.fromarray(img, mode="L").save(out_path)

            print(f"Saved => {out_path}")
            decoded_total += 1

            # stop processing if we have decoded enough images
            if NUM_IMAGES and decoded_total >= NUM_IMAGES:
                break
        print()
    print(f"Decoded {decoded_total} images in total")

if __name__ == "__main__":
    main()