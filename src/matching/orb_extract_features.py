import argparse
import datetime

from classes.orb_feature_extractor import OrbFeatureExtractor

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
output_file_name = args["output"]

if verbose:
    print('Going to write features to a file "{}": {:%H:%M:%S.%f}'.format(output_file_name, datetime.datetime.now()))

feature_extractor = OrbFeatureExtractor(verbose)
extracted_features = feature_extractor.extract(args["images"], args["n_features"])

if verbose:
    print('All features have been extracted, serializing...: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

feature_extractor.serialize(extracted_features, output_file_name)

if verbose:
    print('Done.')
