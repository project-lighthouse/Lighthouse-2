import argparse
import datetime

from classes.feature_extractor import FeatureExtractor

parser = argparse.ArgumentParser(description='Finds, extracts and saves the best features of the provided image set.')
parser.add_argument('-i', '--images', required=True,
                    help='Path to the folder with the images we would like to extract features for.')
parser.add_argument('-o', '--output', required=True,
                    help='Path to the file that will store all extracted features (in JSON format).')
parser.add_argument('--orb-n-features',
                    help='Number of features to extract from every image when ORB detector is used (default: 2000)',
                    default=2000, type=int)
parser.add_argument('--akaze-n-channels',
                    help='Number of channels in the descriptor when AKAZE detector is used (default: 3)',
                    choices=[1, 2, 3], default=3, type=int)
parser.add_argument('--detector', help='Feature detector to use (default: orb)', choices=['orb', 'akaze'],
                    default='orb')
parser.add_argument('--verbose', help='Increase output verbosity', action='store_true')
args = vars(parser.parse_args())

verbose = args["verbose"]
output_file_name = args["output"]

if verbose:
    print('Going to write features to a file "{}": {:%H:%M:%S.%f}'.format(output_file_name, datetime.datetime.now()))

feature_extractor = FeatureExtractor(verbose)

options = dict(orb_n_features=args['orb_n_features'], akaze_n_channels=args['akaze_n_channels'])

extracted_features = feature_extractor.extract(args["images"], args["detector"], options)

if verbose:
    print('All features have been extracted, serializing...: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

feature_extractor.serialize(extracted_features, output_file_name)

if verbose:
    print('Done.')
