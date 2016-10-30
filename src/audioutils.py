from __future__ import division
import math
import time
from array import array
from threading import Thread
import os
import alsaaudio

ALSA_SPEAKER = "plug:default"     # ALSA device identifier
ALSA_MICROPHONE = "plug:default"  # ALSA device identifier
BYTES_PER_SAMPLE = 2
FORMAT = alsaaudio.PCM_FORMAT_S16_LE  # We use signed 16-bit samples
SAMPLES_PER_SECOND = 16000            # at 16000 samples per second

# Adjust this value as needed depending on mic sensitivity.
# For a high-quality USB mic, 150 is a good value.
# For a cheap USB headset, 3000 is better.
# We need to get this value right in order to have recordings
# automatically stop on silence.
DEFAULT_SILENCE_THRESHOLD = 150

# Generate a sine wave of the specified frequency and duration with
# an amplitude that starts high and drops to zero. This does not actually
# play a sound. It just returns an array that can be passed to audio.play()
def makebeep(frequency, duration):
    # Generate the waveform for this beep
    samples = array('h')
    numsamples = int(SAMPLES_PER_SECOND * duration)
    samples_per_cycle = SAMPLES_PER_SECOND / frequency;
    angle_per_sample = 2 * math.pi / samples_per_cycle;
    phase = 0.0;
    for i in range(0, numsamples):
        factor = 1 - (i / numsamples)
        samples.append(int(factor * 32000 * math.sin(phase)))
        phase = (phase + angle_per_sample) % (2*math.pi)

    return samples

# Play the specified audio samples though the speakers.
# This function expects and array or bytes object like those returned
# by the makebeep() and record() functions.
def _play(samples):
    if (len(samples) == 0):
        return
    speaker = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK, card=ALSA_SPEAKER)
    speaker.setchannels(1)
    speaker.setrate(SAMPLES_PER_SECOND)
    speaker.setformat(FORMAT)
    speaker.setperiodsize(len(samples)//BYTES_PER_SAMPLE)
    speaker.write(samples)

def play(samples):
    # the _play() function above only blocks until all the
    # samples are buffered by the kernel, so it may return
    # before the sound has finished playing.
    # We want our function to block until the sound is done
    start = time.time()
    _play(samples)
    end = time.time()
    duration = len(samples)/(BYTES_PER_SAMPLE * SAMPLES_PER_SECOND)
    elapsed = end - start
    if duration > elapsed:
        time.sleep(duration - elapsed)

# Play the sound using a thread so we can return right away
def playAsync(samples):
    Thread(target=lambda: _play(samples)).start()

def playfile(filename):
    try:
        filesize = os.path.getsize(filename)
        with open(filename, 'rb') as f:
            speaker = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK, card=ALSA_SPEAKER)
            speaker.setchannels(1)
            speaker.setrate(SAMPLES_PER_SECOND)
            speaker.setformat(FORMAT)
            speaker.setperiodsize(1000)
            starttime = time.time();
            while True:
                samples = f.read(2000)
                if not samples:
                    break;
                speaker.write(samples)
            duration = filesize/(BYTES_PER_SAMPLE * SAMPLES_PER_SECOND)
            elapsed = time.time() - starttime
            if duration > elapsed:
                time.sleep(duration - elapsed)

    except IOError as err:
        print("IO error: {0}".format(err))


# Record audio from the microphone, trim off leading and trailing silence
# and return an array of the audio samples. If no sound is detected at all
# then the returned array will have a length of zero.
def record(min_duration=1,         # Record at least this many seconds
           max_duration=8,         # But no more than this many seconds
           max_silence=1,          # Stop recording after silence this long
           silence_factor=0.1):    # Define "silence" as this fraction of max

    chunk_duration = 1.0/16        # record in batches this many seconds long
    chunk_size = int(SAMPLES_PER_SECOND * chunk_duration)
    elapsed = 0.0                  # how many seconds recorded so far
    elapsed_silence = 0.0          # how much silence at the end

    # Chunks that are quieter than this count as silence
    # We start with this default value, but may adjust up depending
    # on how loud the user talks
    silence_threshold = DEFAULT_SILENCE_THRESHOLD

    # We're keep track of how loud the recording is, and start with an
    # initial value chosen so that the silence threshold does not decrease
    max_amplitude = int(silence_threshold / silence_factor)

    mic = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, card=ALSA_MICROPHONE)
    mic.setchannels(1)
    mic.setrate(SAMPLES_PER_SECOND)
    mic.setformat(FORMAT)
    mic.setperiodsize(chunk_size)
    recording = array('h')

    while elapsed < max_duration:
        l, chunk = mic.read()
        chunkarray = array('h', chunk)
        recording.extend(chunkarray)

        chunk_duration = len(chunkarray) / SAMPLES_PER_SECOND
        elapsed += chunk_duration
        chunk_max = max(chunkarray)

        # Keep track of how loud the recording gets
        if chunk_max > max_amplitude:
            max_amplitude = chunk_max
            silence_threshold = max_amplitude * silence_factor

        # If this chunk is relatively quiet compared to the max
        # then treat it as silence
        if chunk_max < silence_threshold:
            elapsed_silence += chunk_duration
        else:
            elapsed_silence = 0.0

        # If we've recorded for at least the minimum time and have
        # recorded at least the maximum silence, then stop recording.
        if elapsed >= min_duration and elapsed_silence >= max_silence:
            break

    # trim the silence from the start and end of the recording
    start = 0;
    end = len(recording) - 1
    while(start < len(recording) and recording[start] < silence_threshold):
        start += 1
    while(end > start and recording[end] < silence_threshold):
        end -= 1
    recording = recording[start:end+1]
    return recording;

if __name__ == '__main__':

    from eventloop import EventLoop

    print('longpress to record');
    print('click to play back recording');
    print('doubleclick to quit');
    eventloop = EventLoop()

    recording = None
    start_recording_tone = makebeep(800, .2)
    stop_recording_tone = makebeep(400, .2)

    def button_handler(event, pin):
        global recording
        if pin != 26:
            return
        if event == 'click':
            if recording:
                play(recording)
            else:
                print("long press to make a recording")
        elif event == 'longpress':
            play(start_recording_tone)
            time.sleep(0.1)  # don't pick up any of the tone in the mic
            recording = record()
            play(stop_recording_tone)
        elif event == 'doubleclick':
            eventloop.exit()


    eventloop.monitor_gpio_button(26, button_handler);
    eventloop.loop()
