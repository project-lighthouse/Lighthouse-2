"""Full toolchain to extract moving objects from a video, add them to a database or compare with existing objects from the database."""

import argparse
import cv2
import capture.capture as capture
import matching.matcher as matcher
import os
import sys

parser = argparse.ArgumentParser(description=
"""
Full toolchain to extract moving objects from a video, add them to a database or compare with existing objects from the database.

With default arguments, the toolchain will extract objects from the webcam and compare with the database.
""")

#
# Main interaction.
#

group = parser.add_argument_group(title="Main interaction")
group.add_argument('--add-to-db', metavar='LABEL', help='Add the object to the database with a label (default: do not add to the database).', default=None)

group.add_argument('--match-with-db', help='Compare the object with objects already present in the database, outputting the most likely label.', dest='match_with_db', action='store_true')
group.add_argument('--no-match-with-db', help='Compare the object with objects already present in the database, outputting the most likely label.', dest='match_with_db', action='store_false')
parser.set_defaults(match_with_db=True)

#
# Customizing interactions with the db.
#
group = parser.add_argument_group(title="Database (default values are generally fine)")
group.add_argument('--rebuild-db', help='Rebuild the database.', dest='rebuild_db', action='store_true')
group.add_argument('--no-rebuild-db', help='Do not rebuild the database (default).', dest='rebuild_db', action='store_false')
parser.set_defaults(rebuild_db=False)

group.add_argument('--db-path', help='Path to the database of images (default: ~/.lighthouse/records).', default='~/.lighthouse/records')


#
# Image acquisition from a video source.
#
# Default options should generally be fine, additional options are provided to help with testing.
#

group = parser.add_argument_group(title="Image acquisition (default values are generally fine)")

group.add_argument('--video-source', help='Use this video source for image capture (default: built-in cam).', default=0)
group.add_argument('--image-source', help='Use this image instead of a video source. Can be specified multiple times. Incompatible with --video-source.', action='append')
group.add_argument('--dump-raw-video', help='Write raw captured video to this file (default: none).', default=None)


group.add_argument('--video-acquisition-autostart', help='Start capturing immediately (default).', dest='autostart', action='store_true')
group.add_argument('--no-video-acquisition-autostart', help='Do not start capturing immediately. You\'ll need a keyboard to start processing.', dest='autostart', action='store_false')
parser.set_defaults(autostart=True)

group.add_argument('--video-width', help='Video width for capture (default: 320).', default=320, type=int)
group.add_argument('--video-height', help='Video height for capture (default: 200).', default=200, type=int)

#
# Video stabilization.
# CAVEAT: Highly experimental, you probably shouldn't use it.
#

group = parser.add_argument_group(title="Video stabilization (CAVEAT: doesn't work yet - you should probably leave this alone)")
group.add_argument('--video-stabilize', help='Stabilize video.', dest='stabilize', action='store_true')
group.add_argument('--no-video-stabilize', help='Do not stabilize video (default).', dest='stabilize', action='store_false')
parser.set_defaults(stabilize=False)

parser.add_argument('--dump-stabilized', help='Write stabilized video to this file (default: none)', default=None)


#
# Customizing how images are extracted from videos.
#

group = parser.add_argument_group(title="Acquiring objects from images (default values are generally fine)")
group.add_argument('--acquisition-dump-objects-prefix', help='Write captured objects to this destination (default: none).', default=None, dest='objects_prefix')
group.add_argument('--acquisition-dump-masks-prefix', help='Write captured masks (before preprocessing) to this destination (default: none).', default=None, dest='masks_prefix')
group.add_argument('--acquisition-keep-objects', metavar='N', help='Keep N "best results" (default: 3) from the video source.', default=3, type=int)

group.add_argument('--acquisition-blur', metavar='SIZE', help='Blur radius (default: 15).', default=15, type=int)
group.add_argument('--acquisition-min-size', metavar='SIZE', help='Clean up the acquired image by assuming that everything speckle with fewer than N pixels is a parasite (default: 100).', default=100, type=int)

group.add_argument('--acquisition-buffer-size', metavar='SIZE', help='Capture at least SIZE frames before proceeding (default: 60).', default=60, type=int)
group.add_argument('--acquisition-buffer-stable-frames', metavar='N', help='Wait for at least N *consecutive stable* frames before acquiring image (default: 0).', default=0, type=int)
group.add_argument('--acquisition-buffer-stability', help='Max proportion of the image that can change before we accept that a frame is stable (default: .1)', default=.1, type=float)
group.add_argument('--acquisition-buffer-init', help='Proportion of frames to keep for initializing background elimination, must be in ]0, 1[ (default: .9)', default=.9, type=float)

group.add_argument('--acquisition-fill', help='Attempt to remove holes from the captured image.', dest='fill_holes', action='store_true')
group.add_argument('--no-acquisition-fill', help='Do not attempt to remove holes from the captured image (default).', dest='fill_holes', action='store_false')
parser.set_defaults(fill_holes=False)

group.add_argument('--acquisition-remove-shadows', help='Pixels that look like shadows should not be considered part of the extracted object.', dest='remove_shadows', action='store_true')
group.add_argument('--no-acquisition-remove-shadows', help='Pixels that look like shadows should be considered part of the extracted object (default).', dest='remove_shadows', action='store_false')
parser.set_defaults(remove_shadows=False)

group.add_argument('--acquisition-use-contour', help='Use contours to turn the image into a set of polygons instead of a cloud of points (default).', dest='use_contour', action='store_true')
group.add_argument('--no-acquisition-use-contour',help='Produce a cloud of points.', dest='use_contour', action='store_false')
parser.set_defaults(use_contour=True)

parser.add_argument('--acquisition-dump-contours-prefix', help='Write contours to this destination (default: none).', default=None)

#
# General.
#

group = parser.add_argument_group(title="General options")
group.add_argument('--gui', help='Display videos, expect keyboard interaction (default).', dest='show', action='store_true')
group.add_argument('--no-gui', help='Do not display videos, don''t expect keyboard interaction.', dest='show', action='store_false')
parser.set_defaults(show=True)

group.add_argument('--verbose', help='Increase output verbosity', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)

#
# FIXME: Not integrated yet.
#

group = parser.add_argument_group(title="More options (TODO: integrate these somewhere)")

group.add_argument('-s', '--source', help='Video to use (default: built-in cam)', default=0)
# group = parser.add_mutually_exclusive_group(required=True)
# group.add_argument('-i', '--images', help='Path to the folder with the images we would like to match')
# group.add_argument('-d', '--data', help='Path to the folder with the images we would like to match')
group.add_argument('--detector', help='Feature detector to use (default: orb)', choices=['orb', 'akaze', 'surf'],
                    default='orb')
group.add_argument('--matcher', help='Matcher to use (default: brute-force)', choices=['brute-force', 'flann'],
                    default='brute-force')
group.add_argument('--n-matches', help='Number of best matches to display  (default: 3)', default=3, type=int)
group.add_argument('--ratio-test-k', help='Ratio test coefficient (default: 0.75)', default=0.75, type=float)
group.add_argument('--n-frames', help='How many frames to capture for matching (default: 100)', default=100, type=int)
group.add_argument('--orb-n-features', help='Number of features to extract used in ORB detector (default: 2000)',
                    default=2000, type=int)
group.add_argument('--akaze-n-channels', help='Number of channels used in AKAZE detector (default: 3)',
                    choices=[1, 2, 3], default=3, type=int)
group.add_argument('--surf-threshold',
                    help='Threshold for hessian keypoint detector used in SURF detector (default: 1000)',
                    default=1000, type=int)
group.add_argument('--cmd-ui', help='Use command line interface to manage the app', action='store_true')


#
# Ensure consistency.
#
args = vars(parser.parse_args())
if args['acquisition_buffer_init'] <= 0:
    args['acquisition_buffer_init'] = .01
elif args['acquisition_buffer_init'] >= 1:
    args['acquisition_buffer_init'] = .99
print ("Args: %s" % args)


def main():
    capture.init(args)
    matcher.init(args)

    #
    # We need at least one source image. We can grab it
    #
    # - from the webcam;
    # - from a video file (useful for testing);
    # - from an image file (useful for testing)
    #
    images = None
    if args['image_source']:
        images = []
        for source in args['image_source']:
            images.append(cv2.imread(source))
    else:
        # Capture a video, either from the webcam or from a video file.
        # This also handles stabilization.
        images = capture.acquire(args)

    if args['rebuild_db']:
        matcher.rebuild_db(args)

    if args['add_to_db']:
        matcher.add_captures(images, args['add_to_db'], args)

    if args['match_with_db']:
        results = matcher.find_closest_match(images, args)

    # FIXME: Do something with the results.

if __name__ == '__main__':
    sys.exit(main())

