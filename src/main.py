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

OPTIONS = config.get_config()

# Set up the audio devices if they are configured
if OPTIONS.audio_out_device:
    audioutils.ALSA_SPEAKER = OPTIONS.audio_out_device
if OPTIONS.audio_in_device:
    audioutils.ALSA_MICROPHONE = OPTIONS.audio_in_device

# Define some sounds that we will be playing
START_RECORDING_TONE = audioutils.makebeep(800, .2)
STOP_RECORDING_TONE = audioutils.makebeep(400, .2)

with open('sounds/shutter.raw', 'rb') as f:
    SHUTTER_TONE = f.read()

# Load the database of items we know about.
DB = ImageDatabase(OPTIONS)

# Initialize the camera object we'll use to take pictures.
CAMERA = Camera(OPTIONS.video_source,
                OPTIONS.video_width,
                OPTIONS.video_height,
                OPTIONS.video_fps)

logging.basicConfig(level=logging.DEBUG if OPTIONS.verbose else logging.INFO)


def take_picture():
    audioutils.playAsync(SHUTTER_TONE)
    return CAMERA.capture()


def match_item():
    matches = []
    matches_count = 0

    # Image with the larger number of matches.
    image = None

    # We'll take up to this many pictures in order to find match.
    for _ in range(0, OPTIONS.matching_n_frames):
        image = take_picture()

        # FIXME: That's bad, we should check all frames we have before we fail.
        try:
            matches = DB.match(image)
        except TooFewFeaturesException:
            print('Too few features')
            audioutils.playfile('sounds/nothing_recognized.raw')
            return
        else:
            matches_count = len(matches)
            if matches_count > 0:
                break

    if matches_count == 0:
        audioutils.playfile('sounds/noitem.raw')
    elif matches_count == 1:
        (_, item) = matches[0]
        audioutils.playfile(item.audio_filename())
    else:
        audioutils.playfile('sounds/multipleitems.raw')
        for (_, match) in matches:
            audioutils.playfile(match.audio_filename())
            time.sleep(0.2)

    if OPTIONS.photo_log and matches_count > 0:
        start = time.time()
        filename = "{}/{}.jpeg".format(OPTIONS.photo_log,
                                       time.strftime("%Y%m%dT%H%M%S"))
        (score, item) = matches[0]
        match_image = item.draw_match(image)
        cv2.putText(match_image, "Score: {}".format(score), (10, 25),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv2.imwrite(filename, match_image)
        print("match photo saved in {}s".format(time.time() - start))


def record_new_item():
    image = take_picture()
    try:
        description = ImageDescription.from_image(image)
    except TooFewFeaturesException:
        print('Too few features')
        audioutils.playfile('sounds/nothing_recognized.raw')
        return

    audio = None
    while audio is None:
        audioutils.playfile('sounds/afterthetone.raw')
        audioutils.play(START_RECORDING_TONE)
        audio = audioutils.record()
        audioutils.play(STOP_RECORDING_TONE)
        if len(audio) < 800:  # if we got less than 50ms of sound
            audio = None
            audioutils.playfile('sounds/nosound.raw')

    item = DB.add(image, audio, description)
    print("added image in {}".format(item.dirname))
    audioutils.playfile('sounds/registered.raw')
    audioutils.play(audio)


def button_handler(event):
    if event == 'press':
        CAMERA.start()
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
    # If we are going to be logging photos that the user takes,
    # make sure the log directory exists.
    if OPTIONS.photo_log and not os.path.isdir(OPTIONS.photo_log):
        os.makedirs(OPTIONS.photo_log)

    # If --web-root was specified, run a web server in a separate process
    # to expose the files in that directory.
    # Note that we're using port 80, assuming we'll always run as root.
    if OPTIONS.web_root:
        subprocess.Popen(['python', '-m', 'SimpleHTTPServer', '80'],
                         cwd=OPTIONS.web_root)

    # Monitor the button for events
    eventloop = EventLoop()
    eventloop.monitor_gpio_button(OPTIONS.gpio_pin, button_handler,
                                  doubleclick_speed=0)

    # If you don't have a button, use --cmd-ui to monitor the keyboard instead.
    if OPTIONS.cmd_ui:
        # Print instructions.
        keyboard_handler()
        # Monitor it on the event loop.
        eventloop.monitor_console(keyboard_handler, prompt="Command:")

    # Run the event loop forever
    eventloop.loop()

if __name__ == '__main__':
    main()
