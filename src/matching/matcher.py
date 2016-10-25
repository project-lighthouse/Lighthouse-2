import cv2
import errno
import numpy
import os
import time
import uuid

from classes.feature_extractor import FeatureExtractor
from classes.image_description import ImageDescription

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_detector(detector_type, options):
    if detector_type == 'orb':
        # Initialize the ORB descriptor, then detect keypoints and extract local invariant descriptors from the image.
        detector = cv2.ORB_create(nfeatures=options['orb_n_features'])
        norm = cv2.NORM_HAMMING
    elif detector_type == 'akaze':
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


class Matcher:
    def __init__(self, options):
        self.options = options
        # Initialize the detector
        detector_options = dict(orb_n_features=options['matching_orb_n_features'],
                                akaze_n_channels=options['matching_akaze_n_channels'],
                                surf_threshold=options['matching_surf_threshold'])

        self._options = options
        self._verbose = options['verbose']
        self._detector, norm = get_detector(options['matching_detector'], detector_options)
        self._matcher = get_matcher(options['matching_matcher'], norm)
        self._feature_extractor = FeatureExtractor(self._verbose)
        self._db = None

    def preload_db(self):
        """ Pre-loads the database from directory args['db_path'].
        preload_db() -> ()
        """
        self._db = self._feature_extractor.deserialize(self._options['db_path'])

    def match(self, images):
        """Finds the best match for the provided video frames in loaded database.
            match() -> (dict(score, db_image_description, frame_index, good_matches), frames)"""

        best_match = dict(score=0, db_image_description=None, frame_index=None, good_matches=None)
        ratio_test_coefficient = self._options['matching_ratio_test_k']
        frames = []

        # Capture "n_frames" frames, try to find match for every one and return match the best score.
        for image_index, image in enumerate(images):
            frame_description = self.get_image_description(image)

            # If we couldn't get frame description (aka keypoints), let's skip this frame.
            if frame_description is None:
                continue

            # Remember all processed frames, to choose from in case we can't find a match.
            frames.append({'frame': image, 'frame_description': frame_description})

            # Iterate through entire database and try to find the best match based on the "score" value.
            match_frame_start = time.time()
            for idx, db_image_description in enumerate(self._db):
                matches = self._matcher.knnMatch(frame_description.descriptors, db_image_description.descriptors, k=2)

                # Apply ratio test.
                good_matches = []
                for match in matches:
                    if len(match) == 2 and match[0].distance < ratio_test_coefficient * match[1].distance:
                        good_matches.append([match[0]])

                histogram_comparison_result = cv2.compareHist(frame_description.histogram,
                                                              db_image_description.histogram, cv2.HISTCMP_CORREL)

                matches_count = len(matches)
                if matches_count == 0:
                    score = 0
                else:
                    good_matches_count = len(good_matches)
                    db_image_descriptors_count = len(db_image_description.descriptors)

                    score = (1 - numpy.absolute(len(frame_description.descriptors) - db_image_descriptors_count) /
                             db_image_descriptors_count) * \
                            (good_matches_count / float(matches_count) * 100) + histogram_comparison_result

                if self._verbose:
                    print('Score for %s is %s.' % (db_image_description.key, score))

                if score > best_match['score']:
                    best_match['score'] = score
                    best_match['db_image_description'] = db_image_description
                    best_match['frame_index'] = image_index
                    best_match['good_matches'] = good_matches

            if self._verbose:
                print('Frame %s is processed in %s seconds.' % (image_index, time.time() - match_frame_start))

        best_match = best_match if best_match['score'] > 0 else None

        return best_match, frames

    def get_image_description(self, image):
        """Extracts image's feature keypoints and color histogram.
                    get_image_description(image) -> ImageDescription"""

        histogram_time = time.time()

        # Calculate frame color histogram.
        histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()

        if self._verbose:
            print('Image histogram calculated in %s seconds.' % (time.time() - histogram_time))

        detector_time = time.time()

        # Extract all possible keypoints from the frame.
        (keypoints, descriptors) = self._detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)

        keypoints = keypoints if keypoints is not None else []
        descriptors = descriptors if descriptors is not None else []

        if self._verbose:
            print('Image keypoints (%s) and descriptors (%s) extracted in %s seconds.' %
                  (len(keypoints), len(descriptors), time.time() - detector_time))

        # If can't extract enough keypoints, that means something wrong with the frame (too dark, blurry etc.).
        if len(keypoints) < self._options['matching_keypoints_threshold']:
            return None

        return ImageDescription(descriptors, histogram)

    def add_image_to_db(self, image, key):
        """Extracts image's feature keypoints and color histogram and saves it to the database under specified key. If
        we can't extract enough keypoints from the image, we return `False` to notify user about the issue.
           add_image_to_db(image, key) -> boolean"""

        # First let's extract image features.
        image_description = self.get_image_description(image)

        if image_description is None:
            return False

        # And fill in other required fields
        image_description.key = key
        image_description.sub_key = uuid.uuid4()

        self._db.append(image_description)

        # Serialize features for new image.
        serialize_start = time.time()
        self._feature_extractor.serialize([image_description], self._options['db_path'])
        if self._verbose:
            print('Entry has been serialized in %s seconds.' % (time.time() - serialize_start))

        # Save image itself if --db-store-images is provided.
        if self._options['db_store_images']:
            key_path = '%s/%s' % (self._options['db_path'], image_description.key)
            make_dir(key_path)
            cv2.imwrite('%s/%s.jpg' % (key_path, image_description.sub_key), image)

        return True

    def draw_match(self, match, frame):
        """Draws lines between matched keypoints in the db image and random frame using provided "match" dictionary.
           draw_match(match) -> ()"""
        db_image = cv2.imread('%s/%s/%s.jpg' % (self._options['db_path'], match['db_image_description'].key,
                                                match['db_image_description'].sub_key))
        if db_image is None:
            print('Can not find image for the db image description in --db-path folder!')
            return

        db_image_keypoints = self._detector.detect(cv2.cvtColor(db_image, cv2.COLOR_BGR2GRAY))
        frame_image_keypoints = self._detector.detect(cv2.cvtColor(frame['frame'], cv2.COLOR_BGR2GRAY))

        result_image = cv2.drawMatchesKnn(db_image, db_image_keypoints, frame['frame'], frame_image_keypoints,
                                          match['good_matches'], None, flags=2)
        cv2.imshow('best-match', result_image)
        cv2.waitKey(0)
