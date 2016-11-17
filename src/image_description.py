from __future__ import division

import os
import time
import logging
import numpy as np
import cv2

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6

# These variables specify how we extract and match features from an image
# They're initialized by ImageDescription.init()
feature_extractor = None
feature_matcher = None
ratio_test_k = None
histogram_weight = None
minimum_keypoints = None
logger = None

# This exception is raised if we can't find enough features in an image
class TooFewFeaturesException(Exception):
    pass


class ImageDescription(object):
    feature_extractor = None
    feature_matcher = None
    ratio_test_k = None
    histogram_weight = None
    minimum_keypoints = None

    # Static class initializer method. Variables that are initialized inside
    # specify how we extract and match features from an image.
    @staticmethod
    def init(options):
        if options.matching_detector == 'orb':
            detector = cv2.ORB_create(
                nfeatures=options.matching_orb_n_features)
            norm = cv2.NORM_HAMMING
        elif options.matching_detector == 'akaze':
            detector = cv2.AKAZE_create(
                descriptor_channels=options.matching_akaze_n_channels)
            norm = cv2.NORM_HAMMING
        else:
            detector = cv2.xfeatures2d.SURF_create(
                hessianThreshold=options.matching_surf_threshold)
            norm = cv2.NORM_L2

        if options.matching_matcher == 'brute-force':
            # Create Brute Force matcher.
            matcher = cv2.BFMatcher(norm)
        else:
            if norm == cv2.NORM_HAMMING:
                flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                    table_number=6,
                                    key_size=12,
                                    multi_probe_level=1)
            else:
                flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

            # Create FLANN matcher.
            matcher = cv2.FlannBasedMatcher(flann_params, {})

        global feature_extractor
        feature_extractor = detector
        global feature_matcher
        feature_matcher = matcher
        global ratio_test_k
        ratio_test_k = options.matching_ratio_test_k
        global histogram_weight
        histogram_weight = options.matching_histogram_weight
        global minimum_keypoints
        minimum_keypoints = options.matching_keypoints_threshold
        global logger
        logger = logging.getLogger(__name__)

    # Private constructor. Use one of the factory functions below
    def __init__(self, dirname, features, histogram):
        self.dirname = dirname
        self.features = features
        self.histogram = histogram

    # Factory function that returns an ImageDescription read from the specified
    # directory.
    @staticmethod
    def from_directory(dirname):
        datafile = "{}/{}".format(dirname, "data.npz")
        with np.load(datafile) as data:
            return ImageDescription(dirname,
                                    data['features'],
                                    data['histogram'])

    # Factory function that returns an ImageDescription created from the
    # specified image data. The returned object does not have a dirname
    # or any associated audio data until it is saved with the write() method
    @staticmethod
    def from_image(image_data):
        # Extract all possible keypoints from the frame.
        grayscale = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        mask = cv2.split(image_data)[3]
        (keypoints, features) = feature_extractor.detectAndCompute(grayscale,
                                                                   mask)

        if len(keypoints) < minimum_keypoints:
            raise TooFewFeaturesException()

        # Calculate color histogram.
        histogram = cv2.calcHist([image_data], [0, 1, 2], None,
                                 [8, 8, 8], [0, 256, 0, 256, 0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()

        return ImageDescription(None, features, histogram)

    # This method saves the item description to the specified directory.
    # If the image data and audio data are specified, they are also saved
    def save(self, dirname, audio_data=None, image_data=None):
        self.dirname = dirname
        os.makedirs(dirname)

        datafile = "{}/{}".format(dirname, "data.npz")
        np.savez(datafile, features=self.features, histogram=self.histogram)

        if image_data is not None:
            cv2.imwrite("{}/{}".format(dirname, "image.png"), image_data)

        if audio_data is not None:
            with open("{}/{}".format(dirname, "audio.raw"), "wb") as audio_file:
                audio_file.write(audio_data)

    def audio_filename(self):
        if self.dirname is None:
            return None
        else:
            return "{}/audio.raw".format(self.dirname)

    def image_filename(self):
        if self.dirname is None:
            return None
        else:
            return "{}/image.png".format(self.dirname)

    # Match one image description against another.
    # Call this method on the newly captured image and pass the description of a
    # stored image from the database. The return value is a number specifying
    # how well the images match. Larger numbers are better matches.
    def compare_to(self, other):
        start = time.time()

        # XXX: check that I've got the order right.
        # other.features should be the features of the stored item image
        # self.features should be the features of the new scene.
        # We're trying to figure out if the item appears in the scene
        # so we're looking for matches between the item's features
        # and the scene's features.
        # The order of the first two arguments here should match the
        # order below in draw_match.
        matches = feature_matcher.knnMatch(other.features, self.features, k=2)

        if len(matches) == 0:
            return 0

        # Apply ratio test: if the best match is significantly better than the
        # second best match then we consider it to be a good match.
        # Note that the absolute distance of the matches does not matter just
        # their relative amounts.
        good_matches = 0
        for match in matches:
            if len(match) == 2:
                distance1 = match[0].distance
                distance2 = match[1].distance
                if distance1 < ratio_test_k * distance2:
                    good_matches += 1

        # If the two images have similar numbers of keypoints this number will
        # be high and will increase the score.
        # feature_ratio = 1 - abs(len(self.features) - len(other.features)) /\
        #                     len(other.features)

        # If most of the feature matches are good ones this ratio will be high
        # and will increase the score.
        good_match_ratio = good_matches / len(matches)

        # Both of the numbers above are between 0 and 1. We take their product
        # and multiply by 100 to create a score between 0 and 100. Kind of a
        # match percentage.
        score = good_match_ratio * 100  # feature_ratio * good_match_ratio * 100

        # Now boost the score based on how well the histograms match
        if histogram_weight:
            histogram_correlation = cv2.compareHist(self.histogram,
                                                    other.histogram,
                                                    cv2.HISTCMP_CORREL)
            score += histogram_weight * histogram_correlation

        logger.debug("Comparison has been made in %ss (matches: %s, "
                     "good matches: %s, score: %s)", time.time() - start,
                     len(matches), good_matches, score)

        return score

    # TODO: We should not recalculate features once again here, we should reuse
    # ones that were produced during matching.
    def draw_match(self, scene):
        item = cv2.imread(self.image_filename())
        gray_item = cv2.cvtColor(item, cv2.COLOR_BGRA2GRAY)
        item_keypoints, item_features = feature_extractor.detectAndCompute(
            gray_item, None)

        gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGRA2GRAY)
        scene_keypoints, scene_features = feature_extractor.detectAndCompute(
            gray_scene, None)

        matches = feature_matcher.knnMatch(item_features, scene_features, k=2)

        good_matches = []
        for match in matches:
            if len(match) == 2:
                distance1 = match[0].distance
                distance2 = match[1].distance
                if distance1 < ratio_test_k * distance2:
                    good_matches.append(match[0])

        good_matches = sorted(good_matches, key=lambda x: x.distance)

        # Take top 25 only.
        match_image = cv2.drawMatches(item, item_keypoints, scene,
                                      scene_keypoints, good_matches[:25], None)
        return match_image
