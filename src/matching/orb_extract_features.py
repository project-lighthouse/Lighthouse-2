import datetime
import argparse
import glob
import json
import cv2

parser = argparse.ArgumentParser(description='Finds, extracts and saves the best features of the provided image set.')
parser.add_argument('-i', '--images', required=True,
                    help='Path to the folder with the images we would like to extract features for.')
parser.add_argument('-o', '--output', required=True,
                    help='Path to the file that will store all extracted features (in JSON format).')
parser.add_argument('--n-features', help='Number of features to extract from every image (default: 2000)', default=2000,
                    type=int)
parser.add_argument('--verbose', help='Increase output verbosity', action='store_true')
args = vars(parser.parse_args())

verbose = args["verbose"]

if verbose:
    print('Args parsed: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

# Initialize the ORB descriptor, then detect keypoints and extract local invariant descriptors from the image.
detector = cv2.ORB_create(nfeatures=args["n_features"])

data = []

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

    image_histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    image_histogram = cv2.normalize(image_histogram, image_histogram).flatten()

    if verbose:
        print('{} image\'s histogram calculated: {:%H:%M:%S.%f}'.format(image_path, datetime.datetime.now()))

    data.append({'key': image_path, 'histogram': image_histogram.tolist(), 'descriptors': image_descriptors.tolist()})

if verbose:
    print('All images processed ({} images): {:%H:%M:%S.%f}'.format(len(data), datetime.datetime.now()))

output_file_name = args["output"]

if verbose:
    print('Writing data to a file "{}"...'.format(output_file_name))

with open(output_file_name, 'w') as outfile:
    json.dump(data, outfile)

if verbose:
    print('Done.')
