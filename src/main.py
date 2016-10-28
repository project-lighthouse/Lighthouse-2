"""Full toolchain to extract moving objects from a video, add them to a database or compare with existing objects from the database."""

import sys
import time
import cv2
import logging
import config
from camera import Camera
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

camera = Camera(options.video_source,
                options.video_width,
                options.video_height,
                options.video_fps)

def take_picture():
    audioutils.playAsync(shutter)
    return camera.capture()

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
    audioutils.playfile('sounds/afterthetone.raw')
    audioutils.play(start_recording_tone)
    audio = audioutils.record()
    audioutils.play(stop_recording_tone)
    item = db.add(image, audio)
    print("added image in {}".format(item.dirname))
    audioutils.playfile('sounds/registered.raw')
    audioutils.play(audio)

def button_handler(event, pin):
    if event == 'press':
        camera.start()
    elif event == 'click':
        match_item()
    elif event == 'longpress':
        record_new_item()

# Monitor the button for events
eventloop = EventLoop()
eventloop.monitor_gpio_button(options.gpio_pin, button_handler,
                              doubleclick_speed=0);

#
# If you don't have a button, use --cmd-ui to monitor the keyboard instead
#
if options.cmd_ui:
    def keyboard_handler(s):
        if s == 'R' or s == 'r':
            record_new_item()
        elif s == 'M' or s == 'm':
            match_item()
        elif s == 'Q' or s == 'q':
            sys.exit(0)
        else:
            print('Enter R to record a new item'
                  ' or M to match an item'
                  ' or Q to quit')
    keyboard_handler('') # Print instructions
    # Monitor it on the event loop
    eventloop.monitor_console(keyboard_handler, prompt="Command:")

# Run the event loop forever
eventloop.loop()
