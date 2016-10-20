import argparse
import cv2
import datetime
import os
import sys
import time

from classes.feature_extractor import FeatureExtractor
from classes.video_stream import VideoStream
from classes.image_description import ImageDescription

is_raspberry_pi = os.uname()[1] == 'raspberrypi2'

detector = None
norm = None

# Images stored in the database.
#
# List of ImageDescription
images = []


# Python 2.7 uses raw_input while Python 3 deprecated it in favor of input.
try:
    input = raw_input
except NameError:
    pass

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6


def get_detector(detector_type, options, args):
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


def get_matcher(matcher_type, norm, args):
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


def annotate_images(images, args):
    """Convert images into a processed format supporting fast comparison with other images.
    annotate_images([images], args) -> [statistics], [annotated_images]"""
    statistics = []
    for image in images:
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
        for idx, image_description in enumerate(images):
            images_with_same_key = matcher.knnMatch(template_descriptors, trainDescriptors=image_description.descriptors, k=2)

            if verbose:
                print('{} image\'s match is processed: {:%H:%M:%S.%f}'.format(
                    image_description.key, datetime.datetime.now()))

            # Apply ratio test.
            good_matches = []
            for m in images_with_same_key:
                if len(m) == 2 and m[0].distance < ratio_test_coefficient * m[1].distance:
                    good_matches.append([m[0]])

            if verbose:
                print('{} good matches filtered ({} good matches): {:%H:%M:%S.%f}'.format(image_description.key,
                                                                                          len(good_matches),
                                                                                          datetime.datetime.now()))

            histogram_comparison_result = cv2.compareHist(template_histogram, image_description.histogram,
                                                          cv2.HISTCMP_CORREL)

            if verbose:
                print('{} image\'s histogram difference is calculated: {:%H:%M:%S.%f}'.format(
                    image_description.key, datetime.datetime.now()))
            good_matches_count = len(good_matches)
            matches_count = len(images_with_same_key)

            if matches_count == 0:
                score = 0
            else:
                score = matches_count / float(len(image_description.descriptors)) * \
                        (good_matches_count / float(matches_count) * 100) + histogram_comparison_result

            statistics.append((idx, images_with_same_key, good_matches, histogram_comparison_result, score))
            images.append({'frame': template, 'keypoints': template_keypoints, 'descriptors': template_descriptors,
                       'histogram': template_histogram})

    return statistics, images

def add_captures(captures, key, args):
    """Add images to the database.
    add_images(images, description, args)"""
    images_with_same_key = [x for x in images if x.key == key]

    image_description = ImageDescription(key, len(images_with_same_key), frame['descriptors'],
                                         frame['histogram'])
    images.append(image_description)

    # save image
    data_source_map = args["data_source_map"]
    if data_source_map is not None:
        key_path = "%s/%s" % (data_source_map, image_description.key)
        if not os.path.exists(key_path):
            os.makedirs(key_path)
        cv2.imwrite("%s/%s.png" % (key_path, image_description.index), frame['frame'])

    # FIXME: Also store histograms.

    print("\033[92mImage successfully added\033[0m")

def rebuild_db(args):
    """ Rebuild the database from directory args['images']. """
    matcher = get_matcher(args['matcher'], norm, args)

    feature_extractor = FeatureExtractor(args['verbose'])
    return feature_extractor.extract(args["images"], args['detector'], detector_options)

def load_db(args):
    """ Load the database from directory args['data'].
    load_db(args) -> [ImageDescription]
    """
    return feature_extractor.deserialize(args["data"])

def find_closest_match(images, statistics, args):
    # Sort by the largest number of "good" matches (6th element (zero based index = 5) of the tuple).
    statistics = sorted(statistics, key=lambda arguments: arguments[5], reverse=True)

    print("\033[94mFull matching has been done in %s seconds.\033[0m" % (time.time() - matching_start))

    for idx, (image_index, images_with_same_key, good_matches, histogram_comparison_result, score) in \
            enumerate(statistics[:10]):
        description = images[image_index]

        # Mark in green only `n-matches` first matches.
        print("{}{}: {} - {} - {} - {}\033[0m".format('\033[92m' if idx < number_of_matches else '\033[91m',
                                                      description.key, len(images_with_same_key), len(good_matches),
                                                      histogram_comparison_result, score))

    best_match = None if len(statistics) == 0 else statistics[0]
    best_score = 0 if best_match is None else statistics[0][5]

    if best_score < 5:
        print("Don't know such object (score %s)" % best_score)
    else:
        description = images[best_match[1]]
        print("Known object (score %s) - %s" % (best_score, description.key))
        image = cv2.imread("%s/%s/%s.png" % (data_source_map, description.key, description.index))
        template = frames[best_match[0]]

        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints = detector.detect(gray_image)

            result_image = cv2.drawMatchesKnn(template['frame'], template['keypoints'], image, keypoints,
                                              best_match[3], None, flags=2)
            cv2.imshow("best-match", result_image)
            cv2.waitKey(0)

def init(args):
    # Initialize the detector
    detector_options = dict(orb_n_features=args['orb_n_features'], akaze_n_channels=args['akaze_n_channels'],
                            surf_threshold=args['surf_threshold'])
    detector, norm = get_detector(args['detector'], detector_options, args)


def main(args):
    start = time.time()

    verbose = args["verbose"]
    cmd_ui = args["cmd_ui"]
    number_of_matches = args["n_matches"]
    ratio_test_coefficient = args["ratio_test_k"]
    number_of_frames = args["n_frames"]


    extraction_start = time.time()

    print("\033[94mTraining set has been prepared in %s seconds.\033[0m" % (time.time() - extraction_start))

    statistics = []
    frames = []
    best_match = None
    best_score = -1

    while True:

        if not gpio_ui and not cmd_ui:
            break

    print("\033[94mProgram has been executed in %s seconds.\033[0m" % (time.time() - start))

    vs.stop()

    if args["show"] and len(statistics) > 0:
        if args["data"] is not None:
            print('\033[93mWarning: Displaying of images side-by-side only works if "{}" is based on existing image '
                  'files and created with the same options (--orb-n-features, --akaze-n-channels, --surf-threshold '
                  'etc.)!\033[0m'.format(args["data"]))

        for idx, (template, template_keypoints, description, images_with_same_key, good_matches, histogram_comparison_result,
                  score) in enumerate(statistics[:number_of_matches]):
            image = cv2.imread(description.key)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints = detector.detect(gray_image)

            result_image = cv2.drawMatchesKnn(template, template_keypoints, image, keypoints, good_matches, None,
                                              flags=2)
            cv2.imshow("Best match #" + str(idx + 1), result_image)

        cv2.waitKey(0)


if __name__ == '__main__':
    sys.exit(main())
