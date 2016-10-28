"""Full toolchain to extract moving objects from a video, add them to a database or compare with existing objects from the database."""

import time
import cv2
import logging
import config
import capture
from eventloop import EventLoop
import audioutils
from image_database import ImageDatabase

options = config.getConfig()

logging.basicConfig(level=logging.DEBUG if options.verbose else logging.INFO)
logger = logging.getLogger(__name__)

logger.debug("Options: %s" % options)

# Set up the audio devices if they are configured
if options.audio_out_device:
    audioutils.ALSA_SPEAKER = options.audio_out_device
if options.audio_in_device:
    audioutils.ALSA_MICROPHONE = options.audio_in_device

# Define some sounds that we will be playing
start_recording_tone = audioutils.makebeep(800, .2)
stop_recording_tone = audioutils.makebeep(400, .2)

with open('sounds/shutter.raw', 'rb') as f:
    shutter = f.read()

db = ImageDatabase(options)

def take_picture():
    audioutils.playAsync(shutter)
    return capture.capture(options)

def match_item():
    matched_items = {}

    # We'll take up to this many pictures in order to find match
    for n in range(0, options.matching_n_frames):
        image = take_picture()
        matches = db.match(image)
        if len(matches) > 0:
            break;

    if len(matches) == 0:
        audioutils.playfile('sounds/noitem.raw')
    elif len(matches) == 1:
        (score,item) = matches[0]
        audioutils.playfile(item.audio_filename())
    else:
        audioutils.playfile('sounds/multipleitems.raw')
        for (score,match) in matches:
            audioutils.playfile(match.audio_filename())
            time.sleep(0.2)

def record_new_item():
    image = take_picture()
    audioutils.play(start_recording_tone)
    audio = audioutils.record()
    audioutils.play(stop_recording_tone)
    item = db.add(image, audio)
    print("added image in {}".format(item.dirname))

def button_handler(event, pin):
    if event == 'click':
        match_item()
    elif event == 'longpress':
        record_new_item()

eventloop = EventLoop()
eventloop.monitor_gpio_button(options.gpio_pin, button_handler);
eventloop.loop()
