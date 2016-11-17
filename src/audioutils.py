from __future__ import division, print_function

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
DEFAULT_SILENCE_THRESHOLD = 1000

# Generate a sine wave of the specified frequency and duration with
# an amplitude that starts high and drops to zero. This does not actually
# play a sound. It just returns an array that can be passed to audio.play()
def makebeep(frequency, duration):
    # Generate the waveform for this beep
    samples = array('h')
    numsamples = int(SAMPLES_PER_SECOND * duration)
    samples_per_cycle = SAMPLES_PER_SECOND / frequency
    angle_per_sample = 2 * math.pi / samples_per_cycle
    phase = 0.0
    for i in range(0, numsamples):
        factor = 1 - (i / numsamples)
        samples.append(int(factor * 32000 * math.sin(phase)))
        phase = (phase + angle_per_sample) % (2*math.pi)

    return samples

# Play the specified audio samples though the speakers.
# This function expects and array or bytes object like those returned
# by the makebeep() and record() functions.
def _play(samples):
    if len(samples) == 0:
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
            starttime = time.time()
            while True:
                samples = f.read(2000)
                if not samples:
                    break
                speaker.write(samples)
            duration = filesize/(BYTES_PER_SAMPLE * SAMPLES_PER_SECOND)
            elapsed = time.time() - starttime
            if duration > elapsed:
                time.sleep(duration - elapsed)

    except IOError as err:
        print("IO error: {0}".format(err))


#
# This class accumulates audio samples via its add() method.
# It computes their average through a sliding window and uses that
# to subtract any DC component from the samples. It also keeps
# track of the highest sample it has seen and uses that to define
# a dynamic silence threshold. It has a method to query to see how
# much "silence" is at the end of the recording, which is useful to
# know when to stop recording, and it has a method to return the
# samples as an array, with silence trimmed off the start and the end.
#
class Recording:
    def __init__(self,
                 window_size=256,
                 silence_factor=0.25,
                 silence_threshold=1000):
        self.window_size = window_size
        self.silence_factor = silence_factor
        self.silence_threshold = silence_threshold
        self.samples = array('h')
        self.window = array('h', [0]*self.window_size)
        self.sum = 0

        self.max = int(self.silence_threshold / self.silence_factor)
        self.silent_samples = 0

    def add(self, sample):
        sampleidx = len(self.samples)
        windowidx = sampleidx % self.window_size
        self.sum -= self.window[windowidx]
        self.window[windowidx] = sample
        self.sum += sample

        if sampleidx >= self.window_size:
            average = self.sum // self.window_size
        else:
            average = self.sum // (sampleidx + 1)

        # the average is the dc component of the signal
        # subtract it out
        sample = sample - average
        if sample > 32767:
            sample = 32767
        elif sample < -32768:
            sample = -32768

        self.samples.append(sample)

        if sample > self.max:
            self.max = sample
            self.silence_threshold = sample * self.silence_factor

        if sample < self.silence_threshold:
            self.silent_samples += 1
        else:
            self.silent_samples = 0

    def duration(self):
        return len(self.samples) / SAMPLES_PER_SECOND

    def trailing_silence(self):
        return self.silent_samples / SAMPLES_PER_SECOND

    # trim most of the silence from the start and end of the recording
    # and return the non-silent samples
    def get_audible_samples(self,
                            start_margin=SAMPLES_PER_SECOND//16,
                            end_margin=SAMPLES_PER_SECOND//8):
        start = 0
        end = len(self.samples) - 1
        while start < len(self.samples) and \
              self.samples[start] < self.silence_threshold:
            start += 1
        while end > start and self.samples[end] < self.silence_threshold:
            end -= 1

        # adjust the start and end to allow a bit of silence
        # on both sides. Except not if everything is silence
        if start < end:
            start = max(0, start - start_margin)
            end = end + end_margin

        return self.samples[start:end+1]

# Record audio from the microphone, trim off leading and trailing silence
# and return an array of the audio samples. If no sound is detected at all
# then the returned array will have a length of zero.
def record(min_duration=1,         # Record at least this many seconds
           max_duration=8,         # But no more than this many seconds
           max_silence=1,          # Stop recording after silence this long
           silence_threshold=DEFAULT_SILENCE_THRESHOLD, # Silence is < this
           silence_factor=0.25):   # Or, less than this fraction of max

    chunk_duration = 1.0/16        # record in batches this many seconds long
    chunk_size = int(SAMPLES_PER_SECOND * chunk_duration)

    mic = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, card=ALSA_MICROPHONE)
    mic.setchannels(1)
    mic.setrate(SAMPLES_PER_SECOND)
    mic.setformat(FORMAT)
    mic.setperiodsize(chunk_size)

    recording = Recording(silence_factor=silence_factor,
                          silence_threshold=silence_threshold)

    while True:
        _, chunk = mic.read()
        chunkarray = array('h', chunk)
        for sample in chunkarray:
            recording.add(sample)

        # How long is the recording now?
        duration = recording.duration()

        # If we've reached the maximum time, stop recording
        if duration >= max_duration:
            break

        # If we've recorded for at least the minimum time and have
        # recorded at least the maximum silence, then stop recording.
        if recording.duration() >= min_duration and \
           recording.trailing_silence() >= max_silence:
            break

    return recording.get_audible_samples()

# Write the specified samples in WAV format.
# The samples must be in the format returned by record():
# single-channel, s16_le samples at 16000 samples per second
def savefile(filename, samples):
    header = array('I',
                   b'RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00'
                   b'\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00'
                   b'\x02\x00\x10\x00data\x00\x00\x00\x00')
    bytelen = len(samples) * 2
    header[10] = bytelen
    header[1] = bytelen + 36

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(samples)


# if __name__ == '__main__':
#
#     from eventloop import EventLoop
#
#     print('longpress to record')
#     print('click to play back recording')
#     print('doubleclick to quit')
#     eventloop = EventLoop()
#
#     recording = None
#     start_recording_tone = makebeep(800, .2)
#     stop_recording_tone = makebeep(400, .2)
#
#     def button_handler(event, pin):
#         global recording
#         if pin != 26:
#             return
#         if event == 'click':
#             if recording:
#                 play(recording)
#             else:
#                 print("long press to make a recording")
#         elif event == 'longpress':
#             play(start_recording_tone)
#             time.sleep(0.1)  # don't pick up any of the tone in the mic
#             recording = record()
#             play(stop_recording_tone)
#         elif event == 'doubleclick':
#             eventloop.exit()
#
#
#     eventloop.monitor_gpio_button(26, button_handler)
#     eventloop.loop()
