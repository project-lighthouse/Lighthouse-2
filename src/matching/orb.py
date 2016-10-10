import argparse
import cv2
import datetime

from classes.orb_feature_extractor import OrbFeatureExtractor

parser = argparse.ArgumentParser(
    description='Finds the best match for the input image among the images in the provided folder.')
parser.add_argument('-t', '--template', required=True, help='Path to the image we would like to find match for')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-i', '--images', help='Path to the folder with the images we would like to match')
group.add_argument('-d', '--data', help='Path to the folder with the images we would like to match')

parser.add_argument('--n-features', help='Number of features to extract from template (default: 2000)', default=2000,
                    type=int)
parser.add_argument('--ratio-test-k', help='Ratio test coefficient (default: 0.75)', default=0.75, type=float)
parser.add_argument('--n-matches', help='Number of best matches to display  (default: 3)', default=3, type=int)
parser.add_argument('--verbose', help='Increase output verbosity', action='store_true')
args = vars(parser.parse_args())

verbose = args["verbose"]

if verbose:
    print('Args parsed: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

# Load the image and convert it to grayscale.
template = cv2.imread(args["template"])
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

if verbose:
    print('Template loaded: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

template_histogram = cv2.calcHist([template], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
template_histogram = cv2.normalize(template_histogram, template_histogram).flatten()

if verbose:
    print('Template histogram calculated: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

# Initialize the ORB descriptor, then detect keypoints and extract local invariant descriptors from the image.
detector = cv2.ORB_create(nfeatures=args["n_features"])

# Create Brute Force matcher.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

(template_keypoints, template_descriptors) = detector.detectAndCompute(gray_template, None)

if verbose:
    print('Template\'s features are extracted: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

statistics = []

ratio_test_coefficient = args["ratio_test_k"]

feature_extractor = OrbFeatureExtractor(verbose)

if args["images"] is not None:
    image_descriptions = feature_extractor.extract(args["images"], args["n_features"])
else:
    image_descriptions = feature_extractor.deserialize(args["data"])

# loop over the images to find the template in
for image_description in image_descriptions:
    matches = matcher.knnMatch(template_descriptors, image_description.descriptors, k=2)

    if verbose:
        print('{} image\'s match is processed: {:%H:%M:%S}'.format(image_description.key, datetime.datetime.now()))

    # Apply ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_test_coefficient * n.distance:
            good_matches.append([m])

    if verbose:
        print('{} good matches filtered ({} good matches): {:%H:%M:%S}'.format(image_description.key, len(good_matches),
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

number_of_matches = args["n_matches"]

for idx, (description, matches, good_matches, histogram_comparison_result) in enumerate(statistics):
    # Display only `n-matches` first matches.
    if idx < number_of_matches:
        #result_image = cv2.drawMatchesKnn(template, template_keypoints, image, keypoints, good_matches, None, flags=2)
        # cv2.imshow("Best match #" + str(idx + 1), result_image)
        color = '\033[92m'
    else:
        color = '\033[91m'
    print("{}{}: {} - {} - {}\033[0m".format(color, description.key, len(matches), len(good_matches),
                                             histogram_comparison_result))

cv2.waitKey(0)
