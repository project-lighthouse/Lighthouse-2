import argparse
import cv2
import datetime
import os
import sys
import time

from classes.feature_extractor import FeatureExtractor

is_raspberry_pi = os.uname()[1] == 'raspberrypi2'

if is_raspberry_pi:
    import RPi.GPIO as GPIO

GPIO_NUMBER = 17
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6

parser = argparse.ArgumentParser(
    description='Finds the best match for the input image among the images in the provided folder.')
parser.add_argument('-s', '--source', help='Video to use (default: built-in cam)', default=0)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-i', '--images', help='Path to the folder with the images we would like to match')
group.add_argument('-d', '--data', help='Path to the folder with the images we would like to match')
parser.add_argument('--detector', help='Feature detector to use (default: orb)', choices=['orb', 'akaze', 'surf'],
                    default='orb')
parser.add_argument('--matcher', help='Matcher to use (default: brute-force)', choices=['brute-force', 'flann'],
                    default='brute-force')
parser.add_argument('--n-matches', help='Number of best matches to display  (default: 3)', default=3, type=int)
parser.add_argument('--ratio-test-k', help='Ratio test coefficient (default: 0.75)', default=0.75, type=float)
parser.add_argument('--n-frames', help='How many frames to capture for matching (default: 100)', default=100, type=int)
parser.add_argument('--orb-n-features', help='Number of features to extract used in ORB detector (default: 2000)',
                    default=2000, type=int)
parser.add_argument('--akaze-n-channels', help='Number of channels used in AKAZE detector (default: 3)',
                    choices=[1, 2, 3], default=3, type=int)
parser.add_argument('--surf-threshold',
                    help='Threshold for hessian keypoint detector used in SURF detector (default: 1000)',
                    default=1000, type=int)
parser.add_argument('--verbose', help='Increase output verbosity', action='store_true')
parser.add_argument('--no-ui', help='Increase output verbosity', action='store_true')
parser.add_argument('--buttons', help='Start capturing only on button click (RPi2 only)', action='store_true')
args = vars(parser.parse_args())


def get_detector(detector_type, options):
    if detector_type == 'orb':
        # Initialize the ORB descriptor, then detect keypoints and extract local invariant descriptors from the image.
        detector = cv2.ORB_create(nfeatures=options['orb_n_features'])
        norm = cv2.NORM_HAMMING
    elif args['detector'] == 'akaze':
        detector = cv2.AKAZE_create(descriptor_channels=options['akaze_n_channels'])
        norm = cv2.NORM_HAMMING
    else:
        detector = cv2.xfeatures2d.SURF_create(hessianThreshold=options['surf_threshold'])
        norm = cv2.NORM_L2

    return detector, norm


def get_matcher(matcher_type, norm):
    if matcher_type == 'brute-force':
        # Create Brute Force matcher.
        matcher = cv2.BFMatcher(norm)
    else:
        if norm == cv2.NORM_HAMMING:
            flann_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

        # Create FLANN matcher.
        matcher = cv2.FlannBasedMatcher(flann_params, {})

    return matcher


def main():
    start = time.time()

    print("\033[94mMain function started.\033[0m")

    verbose = args["verbose"]
    buttons = args["buttons"]

    if buttons and not is_raspberry_pi:
        print("\033[91mArgument 'buttons' can only be used on Raspberry Pi 2.\033[0m")
        return -1

    if buttons:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(GPIO_NUMBER, GPIO.IN)

    if verbose:
        print('Args parsed: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

    detector_options = dict(orb_n_features=args['orb_n_features'], akaze_n_channels=args['akaze_n_channels'],
                            surf_threshold=args['surf_threshold'])

    cap = cv2.VideoCapture(args['source'])
    if cap is None or not cap.isOpened():
        print('Error: unable to open video source')
        return -1

    detector, norm = get_detector(args['detector'], detector_options)
    matcher = get_matcher(args['matcher'], norm)

    statistics = []

    ratio_test_coefficient = args["ratio_test_k"]

    feature_extractor = FeatureExtractor(verbose)

    extraction_start = time.time()

    if args["images"] is not None:
        image_descriptions = feature_extractor.extract(args["images"], args['detector'], detector_options)
    else:
        image_descriptions = feature_extractor.deserialize(args["data"])

    print("\033[94mTraining set has been prepared in %s seconds.\033[0m" % (time.time() - extraction_start))

    number_of_frames = args["n_frames"]

    while True:
        while buttons:
            if GPIO.input(GPIO_NUMBER) == 1:
                print("\033[92mButton is pressed, running matching...\033[0m")
                break
            time.sleep(0.05)

        matching_start = time.time()

        while True:
            ret, template = cap.read()

            if not ret:
                print("No frames is available.")
                break

            if len(statistics) > number_of_frames:
                break

            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            if verbose:
                print('Template loaded: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

            template_histogram = cv2.calcHist([template], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            template_histogram = cv2.normalize(template_histogram, template_histogram).flatten()

            if verbose:
                print('Template histogram calculated: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

            (template_keypoints, template_descriptors) = detector.detectAndCompute(gray_template, None)

            if verbose:
                print('Template keypoints have been detected: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

            # loop over the images to find the template in
            for image_description in image_descriptions:
                matches = matcher.knnMatch(template_descriptors, trainDescriptors=image_description.descriptors, k=2)

                if verbose:
                    print('{} image\'s match is processed: {:%H:%M:%S.%f}'.format(
                        image_description.key, datetime.datetime.now()))

                # Apply ratio test.
                good_matches = []
                for m in matches:
                    if len(m) == 2 and m[0].distance < ratio_test_coefficient * m[1].distance:
                        good_matches.append([m[0]])

                if verbose:
                    print('{} good matches filtered ({} good matches): {:%H:%M:%S.%f}'.format(image_description.key,
                                                                                              len(good_matches),
                                                                                              datetime.datetime.now()))

                histogram_comparison_result = cv2.compareHist(template_histogram, image_description.histogram,
                                                              cv2.HISTCMP_CORREL)

                if verbose:
                    print('{} image\'s histogram difference is calculated: {:%H:%M:%S.%f}'.format(image_description.key,
                                                                                                  datetime.datetime.now()))
                good_matches_count = len(good_matches)
                matches_count = len(matches)

                score = (0 if matches_count == 0 else good_matches_count / float(matches_count)) + \
                        (0.01 * histogram_comparison_result)

                statistics.append((template, template_keypoints, image_description, matches, good_matches,
                                   histogram_comparison_result, score))

        if verbose:
            print('All images have been processed: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

        # Sort by the largest number of "good" matches (3th element (zero based index = 2) of the tuple).
        statistics = sorted(statistics, key=lambda arguments: arguments[6], reverse=True)

        print("\033[94mFull matching has been done in %s seconds.\033[0m" % (time.time() - matching_start))

        # Display results
        number_of_matches = args["n_matches"]

        for idx, (template, template_keypoints, description, matches, good_matches, histogram_comparison_result, score) in \
                enumerate(statistics[:10]):
            # Mark in green only `n-matches` first matches.
            print("{}{}: {} - {} - {} - {}\033[0m".format('\033[92m' if idx < number_of_matches else '\033[91m',
                                                          description.key, len(matches), len(good_matches),
                                                          histogram_comparison_result, score))
        if not buttons:
            break
        else:
            statistics = []

    print("\033[94mProgram has been executed in %s seconds.\033[0m" % (time.time() - start))

    cap.release()

    if not args["no_ui"]:
        if args["data"] is not None:
            print('\033[93mWarning: Displaying of images side-by-side only works if "{}" is based on existing image '
                  'files and created with the same options (--orb-n-features, --akaze-n-channels, --surf-threshold '
                  'etc.)!\033[0m'.format(args["data"]))

        for idx, (template, template_keypoints, description, matches, good_matches, histogram_comparison_result, score) \
                in enumerate(statistics[:number_of_matches]):
            image = cv2.imread(description.key)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints = detector.detect(gray_image)

            result_image = cv2.drawMatchesKnn(template, template_keypoints, image, keypoints, good_matches, None,
                                              flags=2)
            cv2.imshow("Best match #" + str(idx + 1), result_image)

        cv2.waitKey(0)


if __name__ == '__main__':
    sys.exit(main())
