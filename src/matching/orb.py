import datetime
import argparse
import glob
import cv2

parser = argparse.ArgumentParser(
    description='Finds the best match for the input image among the images in the provided folder.')
parser.add_argument('-t', '--template', required=True, help='Path to the image we would like to find match for')
parser.add_argument('-i', '--images', required=True, help='Path to the folder with the images we would like to match')
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

# loop over the images to find the template in
for image_path in glob.glob(args["images"] + "/*.jpg"):
    # Load the image, convert it to grayscale.
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if verbose:
        print('{} image loaded: {:%H:%M:%S.%f}'.format(image_path, datetime.datetime.now()))

    (image_keypoints, image_descriptors) = detector.detectAndCompute(gray_image, None)

    if verbose:
        print('{} image\'s features are extracted: {:%H:%M:%S.%f}'.format(image_path, datetime.datetime.now()))

    matches = matcher.knnMatch(template_descriptors, image_descriptors, k=2)

    if verbose:
        print('{} image\'s match is processed: {:%H:%M:%S}'.format(image_path, datetime.datetime.now()))

    # Apply ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_test_coefficient * n.distance:
            good_matches.append([m])

    image_histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    image_histogram = cv2.normalize(image_histogram, image_histogram).flatten()

    if verbose:
        print('{} image\'s histogram is calculated: {:%H:%M:%S.%f}'.format(image_path, datetime.datetime.now()))

    histogram_comparison_result = cv2.compareHist(template_histogram, image_histogram, cv2.HISTCMP_CORREL)

    if verbose:
        print('{} image\'s histogram difference is calculated: {:%H:%M:%S.%f}'.format(image_path,
                                                                                      datetime.datetime.now()))

    statistics.append((image_path, image_keypoints, matches, good_matches, image, histogram_comparison_result))

if verbose:
    print('All images have been processed: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

# Sort by the largest number of "good" matches (4th element (zero based index = 3) of the tuple).
statistics = sorted(statistics, key=lambda arguments: len(arguments[3]), reverse=True)

number_of_matches = args["n_matches"]

for idx, (path, keypoints, matches, good_matches, image, histogram_comparison_result) in enumerate(statistics):
    # Display only `n-matches` first matches.
    if idx < number_of_matches:
        result_image = cv2.drawMatchesKnn(template, template_keypoints, image, keypoints, good_matches, None, flags=2)
        cv2.imshow("Best match #" + str(idx + 1), result_image)
        color = '\033[92m'
    else:
        color = '\033[91m'
    print("{}{}: {} - {} - {}\033[0m".format(color, path, len(matches), len(good_matches), histogram_comparison_result))

cv2.waitKey(0)
