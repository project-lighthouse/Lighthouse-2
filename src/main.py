"""Full toolchain to extract moving objects from a video, add them to a database
or compare with existing objects from the database.
"""
from __future__ import print_function

import logging
import os
import subprocess
import sys
import time
import cv2

import config
import audioutils
from camera import Camera
from eventloop import EventLoop
from image_database import ImageDatabase
from image_description import ImageDescription, TooFewFeaturesException

# Define base and sounds folder paths.
BASE_PATH = os.path.dirname(__file__)
SOUNDS_PATH = os.path.join(BASE_PATH, 'sounds')

# Define some sounds that we will be playing.
START_RECORDING_TONE = audioutils.makebeep(800, .2)
STOP_RECORDING_TONE = audioutils.makebeep(400, .2)
SHUTTER_TONE = None

db = None
camera = None

options = config.get_config()

logging.basicConfig(level=logging.DEBUG if options.verbose else logging.INFO)
logger = logging.getLogger(__name__)


def get_sound(name):
    return os.path.join(SOUNDS_PATH, name)


def take_picture():
    audioutils.playAsync(SHUTTER_TONE)
    return camera.capture()


def match_item():
    matches = []
    matches_count = 0

    # Image with the larger number of matches.
    image = None

    # We'll take up to this many pictures in order to find match.
    for _ in range(0, options.matching_n_frames):
        image = take_picture()

        # FIXME: That's bad, we should check all frames we have before we fail.
        try:
            matches = db.match(image)
        except TooFewFeaturesException:
            print('Too few features')
            audioutils.playfile(get_sound('nothing_recognized.raw'))
            return
        else:
            matches_count = len(matches)
            if matches_count > 0:
                break

    if matches_count == 0:
        audioutils.playfile(get_sound('noitem.raw'))
    elif matches_count == 1:
        (_, item) = matches[0]
        audioutils.playfile(item.audio_filename())
    else:
        audioutils.playfile(get_sound('multipleitems.raw'))
        for (_, match) in matches:
            audioutils.playfile(match.audio_filename())
            time.sleep(0.2)

    if options.photo_log and matches_count > 0:
        start = time.time()
        filename = "{}/{}.jpeg".format(options.photo_log,
                                       time.strftime("%Y%m%dT%H%M%S"))
        (score, item) = matches[0]
        match_image = item.draw_match(image)
        cv2.putText(match_image, "Score: {}".format(score), (10, 25),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv2.imwrite(filename, match_image)
        logger.debug("match photo saved in %s", time.time() - start)


def record_new_item():
    image = take_picture()
    try:
        description = ImageDescription.from_image(image)
    except TooFewFeaturesException:
        print('Too few features')
        audioutils.playfile(get_sound('nothing_recognized.raw'))
        return

    audio = None
    while audio is None:
        audioutils.playfile(get_sound('afterthetone.raw'))
        audioutils.play(START_RECORDING_TONE)
        audio = audioutils.record()
        audioutils.play(STOP_RECORDING_TONE)
        if len(audio) < 800:  # if we got less than 50ms of sound
            audio = None
            audioutils.playfile(get_sound('nosound.raw'))

    item = db.add(image, audio, description)
    print("added image in {}".format(item.dirname))
    audioutils.playfile(get_sound('registered.raw'))
    audioutils.play(audio)


def button_handler(event, pin):
    logger.debug("Pin #%s is activated by '%s' event", pin, event)

    if event == 'press':
        camera.start()
    elif event == 'click':
        match_item()
    elif event == 'longpress':
        record_new_item()


def keyboard_handler(key=None):
    if key == 'R' or key == 'r':
        record_new_item()
    elif key == 'M' or key == 'm':
        match_item()
    elif key == 'Q' or key == 'q':
        sys.exit(0)
    else:
        print('Enter R to record a new item'
              ' or M to match an item'
              ' or Q to quit')


def main():
    # Load the database of items we know about.
    global db
    db = ImageDatabase(options)

    # Initialize the camera object we'll use to take pictures.
    global camera
    camera = Camera(options.video_source,
                    options.video_width,
                    options.video_height,
                    options.video_fps)

    with open(get_sound('shutter.raw'), 'rb') as f:
        global SHUTTER_TONE
        SHUTTER_TONE = f.read()

    # Set up the audio devices if they are configured
    if options.audio_out_device:
        audioutils.ALSA_SPEAKER = options.audio_out_device
    if options.audio_in_device:
        audioutils.ALSA_MICROPHONE = options.audio_in_device

    # If we are going to be logging photos that the user takes,
    # make sure the log directory exists.
    if options.photo_log and not os.path.isdir(options.photo_log):
        os.makedirs(options.photo_log)

    # If --web-server was specified, run a web server in a separate process
    # to expose the files in that directory.
    # Note that we're using port 80, assuming we'll always run as root.
    if options.web_server:
        subprocess.Popen(['python', '-m', 'SimpleHTTPServer', '80'],
                         cwd=options.web_server_root)

    # Monitor the button for events
    eventloop = EventLoop()
    eventloop.monitor_gpio_button(options.gpio_pin, button_handler,
                                  doubleclick_speed=0)

    # If you don't have a button, use --cmd-ui to monitor the keyboard instead.
    if options.cmd_ui:
        # Print instructions.
        keyboard_handler()
        # Monitor it on the event loop.
        eventloop.monitor_console(keyboard_handler, prompt="Command: ")

    # Run the event loop forever
    eventloop.loop()

if __name__ == '__main__':
    main()
