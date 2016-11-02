# pylint: disable=line-too-long
import argparse
import logging.config
import os


# XXX Commented out arguments are for experimental code that is
# not currently in use
def get_config():
    parser = argparse.ArgumentParser(description="Lighthouse prototype")

    #
    # Main interaction.
    #

    # group = parser.add_argument_group(title="Main interaction")
    # group.add_argument('--add-to-db', metavar='LABEL', help='Add the object to the database with a label (default: do not add to the database).', default=None)

    # group.add_argument('--match-with-db', help='Compare the object with objects already present in the database, outputting the most likely label.', dest='match_with_db', action='store_true')
    # group.add_argument('--no-match-with-db', help='Compare the object with objects already present in the database, outputting the most likely label.', dest='match_with_db', action='store_false')
    # parser.set_defaults(match_with_db=True)

    #
    # Customizing interactions with the db.
    #
    group = parser.add_argument_group(title="Database (default values are generally fine)")
    # group.add_argument('--rebuild-db', help='Rebuild the database.', dest='rebuild_db', action='store_true')
    # group.add_argument('--no-rebuild-db', help='Do not rebuild the database (default).', dest='rebuild_db', action='store_false')
    # parser.set_defaults(rebuild_db=False)

    group.add_argument('--db-path',
                       help='Path to the database of image features (default: ~/Lighthouse/Data).',
                       default='~/Lighthouse/Data')
    group.add_argument('--db-store-images', help='Indicates whether we want to store raw images altogether with features.',
                       action='store_true')

    #
    # Image acquisition from a video source.
    #
    # Default options should generally be fine, additional options are provided to help with testing.
    #

    group = parser.add_argument_group(title="Image acquisition (default values are generally fine)")

    group.add_argument('--video-source', help='Use this video source for image capture (default: built-in cam).', default=0)
    group.add_argument('--image-source', help='Use this image instead of a video source. Can be specified multiple times. Incompatible with --video-source.', action='append')
    # group.add_argument('--dump-raw-video', help='Write raw captured video to this file (default: none).', default=None)


    # group.add_argument('--video-acquisition-autostart', help='Start capturing immediately (default).', dest='autostart', action='store_true')
    # group.add_argument('--no-video-acquisition-autostart', help='Do not start capturing immediately. You\'ll need a keyboard to start processing.', dest='autostart', action='store_false')
    # parser.set_defaults(autostart=True)

    group.add_argument('--video-width', help='Video width for capture (default: 640).', default=640, type=int)
    group.add_argument('--video-height', help='Video height for capture (default: 480).', default=480, type=int)
    group.add_argument('--video-fps', help='Video frame rate in FPS. (default: 15).', default=15, type=int)

    #
    # Video stabilization.
    # CAVEAT: Highly experimental, you probably shouldn't use it.
    #

    # group = parser.add_argument_group(title="Video stabilization (CAVEAT: doesn't work yet - you should probably leave this alone)")
    # group.add_argument('--video-stabilize', help='Stabilize video.', dest='stabilize', action='store_true')
    # group.add_argument('--no-video-stabilize', help='Do not stabilize video (default).', dest='stabilize', action='store_false')
    # parser.set_defaults(stabilize=False)

    # parser.add_argument('--dump-stabilized', help='Write stabilized video to this file (default: none)', default=None)


    #
    # Customizing how images are extracted from videos.
    #

    # group = parser.add_argument_group(title="Acquiring objects from images (default values are generally fine)")
    # group.add_argument('--acquisition-dump-objects-prefix', help='Write captured objects to this destination (default: none).', default=None, dest='objects_prefix')
    # group.add_argument('--acquisition-dump-masks-prefix', help='Write captured masks (before preprocessing) to this destination (default: none).', default=None, dest='masks_prefix')
    # group.add_argument('--acquisition-keep-objects', metavar='N', help='Keep N "best results" (default: 3) from the video source.', default=3, type=int)

    # group.add_argument('--acquisition-blur', metavar='SIZE', help='Blur radius (default: 15).', default=15, type=int)
    # group.add_argument('--acquisition-min-size', metavar='SIZE', help='Clean up the acquired image by assuming that everything speckle with fewer than N pixels is a parasite (default: 100).', default=100, type=int)

    # group.add_argument('--acquisition-buffer-size', metavar='SIZE', help='Capture at least SIZE frames before proceeding (default: 60).', default=60, type=int)
    # group.add_argument('--acquisition-buffer-stable-frames', metavar='N', help='Wait for at least N *consecutive stable* frames before acquiring image (default: 0).', default=0, type=int)
    # group.add_argument('--acquisition-buffer-stability', help='Max proportion of the image that can change before we accept that a frame is stable (default: .1)', default=.1, type=float)
    # group.add_argument('--acquisition-buffer-init', help='Proportion of frames to keep for initializing background elimination, must be in ]0, 1[ (default: .9)', default=.9, type=float)

    # group.add_argument('--acquisition-fill', help='Attempt to remove holes from the captured image.', dest='fill_holes', action='store_true')
    # group.add_argument('--no-acquisition-fill', help='Do not attempt to remove holes from the captured image (default).', dest='fill_holes', action='store_false')
    # parser.set_defaults(fill_holes=False)

    # group.add_argument('--acquisition-remove-shadows', help='Pixels that look like shadows should not be considered part of the extracted object.', dest='remove_shadows', action='store_true')
    # group.add_argument('--no-acquisition-remove-shadows', help='Pixels that look like shadows should be considered part of the extracted object (default).', dest='remove_shadows', action='store_false')
    # parser.set_defaults(remove_shadows=False)

    # group.add_argument('--acquisition-use-contour', help='Use contours to turn the image into a set of polygons instead of a cloud of points (default).', dest='use_contour', action='store_true')
    # group.add_argument('--no-acquisition-use-contour',help='Produce a cloud of points.', dest='use_contour', action='store_false')
    # parser.set_defaults(use_contour=True)

    # parser.add_argument('--acquisition-dump-contours-prefix', help='Write contours to this destination (default: none).', default=None)

    #
    # Customizing how images are compared to each other.
    #

    group = parser.add_argument_group(title="Finding objects in the database")

    group.add_argument('--matching-detector', help='Feature detector to use (default: orb)',
                       choices=['orb', 'akaze', 'surf'], default='orb')
    group.add_argument('--matching-matcher', help='Matcher to use (default: brute-force)', choices=['brute-force', 'flann'],
                       default='brute-force')
    group.add_argument('--matching-ratio-test-k', help='Ratio test coefficient (default: 0.75)', default=0.75, type=float)
    group.add_argument('--matching-histogram-weight', help='How much weight to give to histogram correlation when matching images', default=5.0, type=float)
    group.add_argument('--matching-n-frames', help='How many frames to capture for matching (default: 3)', default=3,
                       type=int)
    group.add_argument('--matching-orb-n-features',
                       help='Number of features to extract used in ORB detector (default: 500)', default=500, type=int)
    group.add_argument('--matching-akaze-n-channels', help='Number of channels used in AKAZE detector (default: 3)',
                       choices=[1, 2, 3], default=3, type=int)
    group.add_argument('--matching-surf-threshold',
                       help='Threshold for hessian keypoint detector used in SURF detector (default: 1000)', default=1000,
                       type=int)
    group.add_argument('--matching-score-threshold',
                       help='Minimal matching score threshold below which we consider image as not matched (default: 10)',
                       default=10, type=float)
    group.add_argument('--matching-score-ratio',
                       help='Secondary matches must have a score at least this fraction of the best match (default: 0.5)',
                       default=0.5, type=float)
    group.add_argument('--matching-keypoints-threshold',
                       help='Minimal number of keypoints that should be extracted from the target image to be considered as'
                       'good enough sample. (default: 50)',
                       default=50, type=int)

    #
    # General.
    #

    group = parser.add_argument_group(title="General options")
    # group.add_argument('--gui', help='Display videos, expect keyboard interaction (default).', dest='show', action='store_true')
    # group.add_argument('--no-gui', help='Do not display videos, don''t expect keyboard interaction.', dest='show', action='store_false')
    # parser.set_defaults(show=True)
    group.add_argument('--cmd-ui', help='Use command line interface to manage the app', action='store_true')

    group.add_argument('--verbose', help='Increase output verbosity', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)

    group.add_argument('--gpio-pin',
                       help='What GPIO pin the button is attached to',
                       default=26, type=int)
    group.add_argument('--audio-out-device',
                       help='The ALSA device name for the speaker',
                       default='plughw:1')
    group.add_argument('--audio-in-device',
                       help='The ALSA device name for the microphone',
                       default='plughw:1')

    group.add_argument('--log-path',
                       help='Directory where all possible logs are stored (default: ~/Lighthouse/Log)',
                       default='~/Lighthouse/Log')

    group.add_argument('--web-server',
                       help='Indicates whether we want to run web server on device (default: false).',
                       action='store_true')
    group.add_argument('--web-server-root',
                       help='Root directory for debugging web server (default: ~/Lighthouse)',
                       default='~/Lighthouse')

    args = parser.parse_args()

    #
    # Ensure consistency.
    #
    # if args.acquisition_buffer_init <= 0:
    #     args.acquisition_buffer_init = .01
    # elif args.acquisition_buffer_init >= 1:
    #     args.acquisition_buffer_init = .99

    # Expand user- and relative-paths.
    args.db_path = os.path.abspath(os.path.expanduser(args.db_path))

    if args.log_path:
        args.log_path = os.path.abspath(os.path.expanduser(args.log_path))

        if not os.path.isdir(args.log_path):
            os.makedirs(args.log_path)

    if args.web_server_root:
        args.web_server_root = os.path.abspath(
            os.path.expanduser(args.web_server_root))

    # Setup logging.
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(module)s:%(levelname)s %(message)s'
            },

            'detailed': {
                'format': '%(asctime)s:%(module)s:%(levelname)s %(message)s'
            }
        },
        'handlers': {
            'default': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            },
            'file': {
                # Once max size is reached, log file will be overridden.
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'detailed',
                # Max 1GB log.
                'maxBytes': 1048576000,
                'filename': os.path.join(args.log_path, 'log.log'),
                'backupCount': 5,
                'encoding': 'utf8'
            }
        },
        'loggers': {
            '': {
                'handlers': ['default', 'file'],
                'level': 'DEBUG' if args.verbose else 'INFO',
                'propagate': True
            }
        }
    })

    return args
