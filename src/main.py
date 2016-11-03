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
from changed_region import get_changed_region

# Define base and sounds folder paths.
BASE_PATH = os.path.dirname(__file__)
SOUNDS_PATH = os.path.join(BASE_PATH, 'sounds')

# Define some sounds that we will be playing.
START_RECORDING_TONE = audioutils.makebeep(800, .2)
STOP_RECORDING_TONE = audioutils.makebeep(400, .2)
CHIRP = audioutils.makebeep(600, .05)
SHUTTER_TONE = None

db = None
camera = None
options = config.get_config()

logger = logging.getLogger(__name__)

# Keep track of whether we're currently busy or not
busy = False

# The EventLoop object
eventloop = None

def get_sound(name):
    return os.path.join(SOUNDS_PATH, name)


def take_picture():
    audioutils.playAsync(SHUTTER_TONE)
    return camera.capture()


def pick_only_accurate_matches(matches):
    # Loop though the scores until we find one that is bigger than the
    # threshold, or significantly bigger than the best score and then return
    # all the matches above that one.
    retval = []
    best_score = matches[0][0] if len(matches) > 0 else 0
    if best_score >= options.matching_score_threshold:
        retval.append(matches[0])
        for match in matches[1:]:
            if match[0] >= options.matching_score_threshold and match[0] >= \
                            best_score * options.matching_score_ratio:
                retval.append(match)
            else:
                break

    return retval


def match_item():
    # Warm up the camera and let it do its white balance during the countdown
    camera.start()

    # "Three...two...one..."
    audioutils.playfile(get_sound('countdown.wav'))

    matches = []

    # Image with the larger number of matches.
    image = None

    # We'll take up to this many pictures in order to find match.
    for _ in range(0, options.matching_n_frames):
        image = take_picture()

        # FIXME: That's bad, we should check all frames we have before we fail.
        try:
            matches = db.match(image)
        except TooFewFeaturesException:
            logger.info("Too few features.")
            audioutils.playfile(get_sound('nothing_recognized.wav'))
            return
        else:
            # Once we find first accurate match, let's stop trying to find more.
            if len(matches) > 0 and matches[0][0] >= \
                    options.matching_score_threshold:
                break

    accurate_matches = pick_only_accurate_matches(matches)

    if len(accurate_matches) == 0:
        audioutils.playfile(get_sound('noitem.wav'))
        logger.debug("No accurate matches found (closest match has score %s).",
                     matches[0][0] if len(matches) > 0 else 0)
    elif len(accurate_matches) == 1:
        (score, item) = accurate_matches[0]
        logger.debug("Found one match with score '%s'.", score)
        audioutils.playfile(item.audio_filename())
    else:
        audioutils.playfile(get_sound('multipleitems.wav'))
        logger.debug("Found several matches with the following scores:")
        for (score, match) in accurate_matches:
            logger.debug("Score: %s", score)
            audioutils.playfile(match.audio_filename())
            time.sleep(0.2)

    if options.log_path and len(matches) > 0:
        start = time.time()
        filename = "{}/{}.jpeg".format(options.log_path,
                                       time.strftime("%Y%m%dT%H%M%S"))
        (score, item) = matches[0]
        match_image = item.draw_match(image)
        cv2.putText(match_image, "Score: {}".format(score), (10, 25),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv2.imwrite(filename, match_image)
        logger.debug("Match photo saved in %s", time.time() - start)


def record_new_item():
    # Make several pictures and choose the frame with the highest number of
    # features. Sometimes camera needs more time to automatically adjust itself
    # for the current light conditions.
    # best_description = None

    # audioutils.playAsync(SHUTTER_TONE)

    # for i in range(0, options.matching_n_frames):
    #     image = take_picture()
    #     try:
    #         description = ImageDescription.from_image(image)

    #         if best_description is None or len(best_description.features) < \
    #                 len(description.features):
    #             best_description = description
    #     except TooFewFeaturesException:
    #         logger.info("Too few features in the frame #%s.", i)

    # if best_description is None:
    #     audioutils.playfile(get_sound('nothing_recognized.wav'))
    #     return

    # Warm up the camera and let it do its white balance while
    # we give the instructions
    camera.start()

    audioutils.playfile(get_sound('register_step1.wav'))
    full_image = take_picture()

    audioutils.playfile(get_sound('register_step2.wav'))
    background_image = take_picture()

    # FIXME: figure out how this function can fail, and handle those cases
    # What happens if we pass two identical images, for example?
    item_image = get_changed_region(full_image, background_image)

    # Check that the returned image isn't too tiny
    # FIXME: this should probably be a different message than
    # the one we play when no features are detected
    if item_image.shape[0] * item_image.shape[1] < 2500:
        audioutils.playfile(get_sound('nothing_recognized.wav'))
        return

    try:
        description = ImageDescription.from_image(item_image)
    except TooFewFeaturesException:
        logger.info("Too few features detected")
        audioutils.playfile(get_sound('nothing_recognized.wav'))
        return

    audio = None
    while audio is None:
        audioutils.playfile(get_sound('afterthetone.wav'))
        audioutils.play(START_RECORDING_TONE)
        audio = audioutils.record()
        audioutils.play(STOP_RECORDING_TONE)
        if len(audio) < 800:  # if we got less than 50ms of sound
            audio = None
            audioutils.playfile(get_sound('nosound.wav'))

    item = db.add(item_image, audio, description)
    logger.info("Added image in %s.", item.dirname)
    audioutils.playfile(get_sound('registered.wav'))
    audioutils.play(audio)

# Play a sound that indicates that Lighthouse is ready for another button press
def ready():
    global busy
    busy = False
    audioutils.play(CHIRP)

def button_handler(event, pin):
    global busy

    # If we're still processing some other event, ignore this one
    if busy:
        logger.debug('ignoring event %s', event)
        return

    logger.debug("Pin #%s is activated by '%s' event", pin, event)

    if event == 'click':
        busy = True
        match_item()
        # run this on the event loop so we ignore any events queueed up
        eventloop.later(ready, 0.5)
    elif event == 'longpress':
        busy = True
        record_new_item()
        eventloop.later(ready, 0.5)


def keyboard_handler(key=None):
    global busy

    if busy:
        logger.debug('ignoring key %s', key)
        return

    if key == 'R' or key == 'r':
        busy = True
        record_new_item()
        eventloop.later(ready, 0.5)
    elif key == 'M' or key == 'm':
        busy = True
        match_item()
        eventloop.later(ready, 0.5)
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

    # If log path is set, make sure the corresponding directory exists.
    if options.log_path and not os.path.isdir(options.log_path):
        os.makedirs(options.log_path)

    # If --web-server was specified, run a web server in a separate process
    # to expose the files in that directory.
    # Note that we're using port 80, assuming we'll always run as root.
    if options.web_server:
        subprocess.Popen(['python', '-m', 'SimpleHTTPServer', '80'],
                         cwd=options.web_server_root)

    # Monitor the button for events
    global eventloop
    eventloop = EventLoop()
    eventloop.monitor_gpio_button(options.gpio_pin, button_handler,
                                  doubleclick_speed=0)

    # If you don't have a button, use --cmd-ui to monitor the keyboard instead.
    if options.cmd_ui:
        # Print instructions.
        keyboard_handler()
        # Monitor it on the event loop.
        eventloop.monitor_console(keyboard_handler, prompt="Command: ")

    # Let the user know we're ready
    ready()

    # Run the event loop forever
    eventloop.loop()

if __name__ == '__main__':
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception:
        logger.exception("Program crashed")
        raise
