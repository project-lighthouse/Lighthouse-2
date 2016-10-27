"""Full toolchain to extract moving objects from a video, add them to a database or compare with existing objects from the database."""

import cv2
import config
import capture
import logging
from matching.matcher import Matcher
import os
import sys
import time

options = config.getConfig()

logging.basicConfig(level=logging.DEBUG if options.verbose else logging.INFO)
logger = logging.getLogger(__name__)

logger.debug("Options: %s" % options)

# Python 2.7 uses raw_input while Python 3 deprecated it in favor of input.
try:
    input = raw_input
except NameError:
    pass


def process_command(command, matcher):
    if command == '1':
        if input('Do you want to add current frame to db (y/n)? > ') == 'n':
            return

        #
        # We need at least one source image. We can grab it
        #
        # - from the webcam;
        # - from a video file or remote stream (useful for testing);
        # - from an image file (useful for testing).
        #
        if options.image_source:
            images = []
            for source in options.image_source:
                images.append(cv2.imread(source))
        else:
            # Capture a video, either from the webcam or from a video file. This also handles stabilization.
            acquire_time = time.time()
            images = capture.capture(options)  # capture.acquire(options)

            logger.debug('Images (%s) have been acquired in %s seconds.' % (len(images), time.time() - acquire_time))

        if matcher.remember_image(images[0], input('Please enter object name > ')):
            print('Image successfully added to the database!')
        else:
            print('\033[91mSorry, we could not add image to the database (not enough light, blurry etc.). '
                  'Please, try again...\033[0m')

        return

    # Let's match currently available frames against our database, choose the frame with best score and if it's larger
    # than "--matching-score-threshold", play the corresponding voice label.
    # Otherwise we should say that we can't recognize the picture and ask if user wants to enter into acquisition mode.
    if command == '2':
        # First acquire images.
        acquire_time = time.time()
        images = capture.capture(options)

        logger.debug('Images (%s) have been acquired in %s seconds.' % (len(images), time.time() - acquire_time))

        # Go through matching process
        match_start = time.time()
        match, image_descriptions, images = matcher.match(images)

        logger.debug('Matching has been done in %s seconds.' % (time.time() - match_start))

        # Not having description for any image means that we couldn't find any good image sample, let's notify user.
        first_good_frame = next((description for description in image_descriptions if description is not None), None)
        if first_good_frame is None:
            print('\033[91mSorry, we could not add image to the database (not enough light, blurry etc.). '
                  'Please, try again...\033[0m')
            return

        if match is None or match['score'] < options.matching_score_threshold:
            if match is None:
                print('\033[91mSorry I can not recognize this object at all :/\033[0m')
            else:
                print('\033[91mSorry I can not recognize this object :/ Closest match is "%s" with score "%s".\033[0m'
                      % (match['description'].key, match['score']))

            print('\033[92mType "1" to add current object to database.\033[0m')
        else:
            print('Object is recognized: %s - %s' % (match['description'].key, match['score']))
            if options.show:
                matcher.draw_match(match, images[match['image_index']])

        return 0

    # FIXME: Awful exit case handling...
    if command == '3':
        return -1

    print('Unknown command.')


def main():
    matcher = Matcher(options)

    logger.debug('Loading matching db...')
    matcher.preload_db()

    cmd_ui = options.cmd_ui

    while True:
        # Here we should wait for the user action via button or console command.
        if cmd_ui:
            command_result = process_command(input('> '), matcher)
            if command_result == -1:
                break
        else:
            print('GPIO support is not implemented yet!')
            return -1

if __name__ == '__main__':
    sys.exit(main())

