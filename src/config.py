# pylint: disable=line-too-long
import argparse
import logging.config
import os


# XXX Commented out arguments are for experimental code that is
# not currently in use
def get_config():
    parser = argparse.ArgumentParser(description="Lighthouse prototype")

    #
    # Customizing interactions with the db.
    #
    group = parser.add_argument_group(title="Database (default values are generally fine)")

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

    group = parser.add_argument_group(title="Image acquisition (default values are generally fine).")

    group.add_argument('--video-source', help='Use this video source for image capture (default: built-in cam).', default=0)
    group.add_argument('--image-source', help='Use this image instead of a video source. Can be specified multiple times. Incompatible with --video-source.', action='append')

    group.add_argument('--video-width', help='Video width for capture (default: 640).', default=640, type=int)
    group.add_argument('--video-height', help='Video height for capture (default: 480).', default=480, type=int)
    group.add_argument('--video-fps', help='Video frame rate in FPS (default: 15).', default=15, type=int)
    group.add_argument('--video-resample-factor', help='Resampling factor to apply before motion detection (default: .3). Lower values increase speed but decrease quality.', default=.3, type=float)

    group.add_argument('--motion-background-removal-strategy', help='Strategy for removing the background (default: "keep-everything"). Can be "keep-everything" (don\'t remove the background), "now-you-see-me" (take a picture without the object then with the object) or "moving-object" (move the object in front of the camera for a second).', default='keep-everything', choices=['keep-everything', 'now-you-see-me', 'moving-object'])
    group.add_argument('--motion-stability-factor', metavar='S', help='Determine when two consecutive frames are considered stable. We check if ||(frame_1, frame_2)|| / surface <= S. (default: .1)', default=.1, type=float)
    group.add_argument('--motion-stability-duration', help='Number of successive stable frames before we assume that the user has stopped moving the object (default: 5).', default=5, type=int)
    group.add_argument('--motion-blur-radius', default=25, type=int)
    group.add_argument('--motion-skip-frames', help='Number of frames we should skip to let background extraction initialize itself properly (default: 20).', default=20, type=int)
    group.add_argument('--motion-discard-small-polygons', metavar='MIN_FRACTION', help='Discard polygons whose pixel surface is smaller than MIN_FRACTION (default: .1).', default=.1, type=float)

    #
    # Customizing how images are compared to each other.
    #

    group = parser.add_argument_group(title="Finding objects in the database")

    group.add_argument('--matching-detector', help='Feature detector to use (default: orb)',
                       choices=['orb', 'akaze', 'surf'], default='orb')
    group.add_argument('--matching-matcher', help='Matcher to use (default: brute-force)', choices=['brute-force', 'flann'],
                       default='brute-force')
    group.add_argument('--matching-ratio-test-k', help='Ratio test coefficient (default: 0.8)', default=0.8, type=float)
    group.add_argument('--matching-histogram-weight', help='How much weight to give to histogram correlation when matching images', default=5.0, type=float)
    group.add_argument('--matching-n-frames', help='How many frames to capture for matching (default: 10)', default=10,
                       type=int)
    group.add_argument('--matching-orb-n-features',
                       help='Number of features to extract used in ORB detector (default: 1000)', default=1000, type=int)
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
    group.add_argument('--max-record-time',
                       help='Max seconds to record item names (default: 5.0)',
                       default=5.0, type=float)
    group.add_argument('--silence-threshold',
                       help='mic levels below this value are treated as silence (default: 1000)',
                       default=1000, type=int)
    group.add_argument('--silence-factor',
                       help='mic levels below this fraction of the highest levels seen are also treated as silence (default: 0.25)',
                       default=0.25, type=float)


    args = parser.parse_args()

    #
    # Ensure consistency.

    if args.video_resample_factor <= 0:
        args.video_resample_factor = .01
    elif args.video_resample_factor > 1:
        args.video_resample_factor = 1
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
