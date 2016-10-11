import argparse
import cv2
import datetime
import time

from classes.feature_extractor import FeatureExtractor

FLANN_INDEX_LSH = 6

start = time.time()

parser = argparse.ArgumentParser(
    description='Finds the best match for the input image among the images in the provided folder.')
parser.add_argument('-t', '--template', required=True, help='Path to the image we would like to find match for')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-i', '--images', help='Path to the folder with the images we would like to match')
group.add_argument('-d', '--data', help='Path to the folder with the images we would like to match')
parser.add_argument('--detector', help='Feature detector to use (default: orb)', choices=['orb', 'akaze'],
                    default='orb')
parser.add_argument('--matcher', help='Matcher to use (default: brute-force)', choices=['brute-force', 'flann'],
                    default='brute-force')
parser.add_argument('--n-matches', help='Number of best matches to display  (default: 3)', default=3, type=int)
parser.add_argument('--ratio-test-k', help='Ratio test coefficient (default: 0.75)', default=0.75, type=float)
parser.add_argument('--orb-n-features',
                    help='Number of features to extract from every image when ORB detector is used (default: 2000)',
                    default=2000, type=int)
parser.add_argument('--akaze-n-channels',
                    help='Number of channels in the descriptor when AKAZE detector is used (default: 3)',
                    choices=[1, 2, 3], default=3, type=int)
parser.add_argument('--verbose', help='Increase output verbosity', action='store_true')
parser.add_argument('--no-ui', help='Increase output verbosity', action='store_true')
args = vars(parser.parse_args())

verbose = args["verbose"]

if verbose:
    print('Args parsed: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

template_start = time.time()

# Load the image and convert it to grayscale.
template = cv2.imread(args["template"])
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

if verbose:
    print('Template loaded: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

template_histogram = cv2.calcHist([template], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
template_histogram = cv2.normalize(template_histogram, template_histogram).flatten()

if verbose:
    print('Template histogram calculated: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

detector_options = dict(orb_n_features=args['orb_n_features'], akaze_n_channels=args['akaze_n_channels'])

if args['detector'] == 'orb':
    # Initialize the ORB descriptor, then detect keypoints and extract local invariant descriptors from the image.
    detector = cv2.ORB_create(nfeatures=detector_options['orb_n_features'])
else:
    detector = cv2.AKAZE_create(descriptor_channels=detector_options['akaze_n_channels'])

if args['matcher'] == 'brute-force':
    # Create Brute Force matcher.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
else:
    # Create FLANN matcher.
    flann_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6,  key_size=12,  multi_probe_level=1)
    matcher = cv2.FlannBasedMatcher(flann_params, {})

(template_keypoints, template_descriptors) = detector.detectAndCompute(gray_template, None)

if verbose:
    print('Template keypoints have been detected: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

print("\033[94mTemplate has been prepared in %s seconds.\033[0m" % (time.time() - template_start))

statistics = []

ratio_test_coefficient = args["ratio_test_k"]

feature_extractor = FeatureExtractor(verbose)

extraction_start = time.time()

if args["images"] is not None:
    image_descriptions = feature_extractor.extract(args["images"], args['detector'], detector_options)
else:
    image_descriptions = feature_extractor.deserialize(args["data"])

print("\033[94mTraining set has been prepared in %s seconds.\033[0m" % (time.time() - extraction_start))

# loop over the images to find the template in
for image_description in image_descriptions:
    matches = matcher.knnMatch(template_descriptors, trainDescriptors=image_description.descriptors, k=2)

    if verbose:
        print('{} image\'s match is processed: {:%H:%M:%S.%f}'.format(image_description.key, datetime.datetime.now()))

    # Apply ratio test.
    good_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < ratio_test_coefficient * m[1].distance:
            good_matches.append([m[0]])

    if verbose:
        print('{} good matches filtered ({} good matches): {:%H:%M:%S.%f}'.format(image_description.key,
                                                                                  len(good_matches),
                                                                                  datetime.datetime.now()))

    histogram_comparison_result = cv2.compareHist(template_histogram, image_description.histogram, cv2.HISTCMP_CORREL)

    if verbose:
        print('{} image\'s histogram difference is calculated: {:%H:%M:%S.%f}'.format(image_description.key,
                                                                                      datetime.datetime.now()))

    statistics.append((image_description, matches, good_matches, histogram_comparison_result))

if verbose:
    print('All images have been processed: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

# Sort by the largest number of "good" matches (3th element (zero based index = 2) of the tuple).
statistics = sorted(statistics, key=lambda arguments: len(arguments[2]), reverse=True)

print("\033[94mFull matching has been done in %s seconds.\033[0m" % (time.time() - start))

# Display results

number_of_matches = args["n_matches"]

for idx, (description, matches, good_matches, histogram_comparison_result) in enumerate(statistics):
    # Mark in green only `n-matches` first matches.
    print("{}{}: {} - {} - {}\033[0m".format('\033[92m' if idx < number_of_matches else '\033[91m', description.key,
                                             len(matches), len(good_matches), histogram_comparison_result))

if not args["no_ui"]:
    if args["data"] is not None:
        print('\033[93mWarning: Displaying of images side-by-side only works if "{}" is based on existing image files '
              'and created with the same options '
              '(--orb-n-features, --akaze-n-channels etc.)!\033[0m'.format(args["data"]))

    for idx, (description, matches, good_matches, histogram_comparison_result) in enumerate(
            statistics[:number_of_matches]):
        image = cv2.imread(description.key)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(gray_image)

        result_image = cv2.drawMatchesKnn(template, template_keypoints, image, keypoints, good_matches, None, flags=2)
        cv2.imshow("Best match #" + str(idx + 1), result_image)

    cv2.waitKey(0)
