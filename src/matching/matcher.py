import cv2
import numpy
import os
import time
from threading import Thread

from classes.feature_extractor import FeatureExtractor
from classes.video_stream import VideoStream
from classes.image_description import ImageDescription

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6


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


def serialize_db(db, feature_extractor, file_name, verbose):
    serialize_start = time.time()
    feature_extractor.serialize(db, file_name)
    if verbose:
        print('Database has been serialized in %s seconds.' % (time.time() - serialize_start))


class Matcher:
    def __init__(self, options):
        self.options = options
        # Initialize the detector
        detector_options = dict(orb_n_features=options['orb_n_features'], akaze_n_channels=options['akaze_n_channels'],
                                surf_threshold=options['surf_threshold'])

        self._options = options
        self._verbose = options['verbose']
        self._detector, norm = get_detector(options['find_detector'], detector_options)
        self._matcher = get_matcher(options['find_matcher'], norm)
        self._feature_extractor = FeatureExtractor(options['verbose'])
        self._db = None

    def preload_db(self):
        """ Pre-loads the database from directory args['db_path'].
        preload_db() -> ()
        """
        self._db = self._feature_extractor.deserialize(self._options['db_path'])

    def match(self):
        """Finds the best match for the provided video frames in loaded database.
            match() -> dict(score, db_image_description, frame, good_matches)"""

        ratio_test_coefficient = self._options['ratio_test_k']

        stream_start = time.time()

        vs = VideoStream(self._options['source']).start()

        if not vs.is_opened():
            print('Error: unable to open video source')
            return -1

        if self._verbose:
            print('Video stream has been prepared in %s seconds.' % (time.time() - stream_start))

        best_match = dict(score=0, db_image_description=None, frame=None, good_matches=None)
        match_all_start = time.time()

        # Capture "n_frames" frames, try to find match for every one and return match the best score.
        for frame_index in range(0, self._options['n_frames']):
            ret, frame = vs.read()

            if not ret:
                print('No frames is available.')
                break

            frame_description = self.get_image_description(frame)

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

                if score > best_match['score']:
                    best_match['score'] = score
                    best_match['db_image_description'] = db_image_description
                    best_match['frame'] = frame
                    best_match['good_matches'] = good_matches

            if self._verbose:
                print('Frame %s is processed in %s seconds.' % (frame_index, time.time() - match_frame_start))

        if self._verbose:
            print('All frames are processed in %s seconds.' % (time.time() - match_all_start))

        vs.stop()

        return best_match

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

        if self._verbose:
            print('Image keypoints extracted in %s seconds.' % (time.time() - detector_time))

        return ImageDescription(descriptors, histogram)

    def add_image_to_db(self, image, key):
        """Extracts image's feature keypoints and color histogram and saves it to the database under specified key.
           add_image_to_db(image, key) -> ()"""

        # First let's extract image features.
        image_description = self.get_image_description(image)

        # Then let's find other existing images for this key.
        images_with_same_key = [x for x in self._db if x.key == key]

        # And fill in other required fields
        image_description.key = key
        image_description.index = len(images_with_same_key)

        self._db.append(image_description)

        # Serialize data to the disk.
        Thread(target=serialize_db,
               args=(self._db, self._feature_extractor, self._options['db_path'], self._verbose)).start()

        # Save image itself if --data-source-map is provided.
        data_source_map = self._options['data_source_map']
        if data_source_map is not None:
            key_path = '%s/%s' % (data_source_map, image_description.key)
            if not os.path.exists(key_path):
                os.makedirs(key_path)
            cv2.imwrite('%s/%s.jpg' % (key_path, image_description.index), image)

    def draw_match(self, match):
        """Draws lines between matched keypoints in the db image and random frame using provided "match" dictionary.
           draw_match(match) -> ()"""
        data_source_map = self._options['data_source_map']
        if data_source_map is None:
            print('Required --data-source-map parameter is not provided, so match can not be drawn!')
            return

        db_image = cv2.imread('%s/%s/%s.jpg' % (data_source_map, match['db_image_description'].key,
                                                match['db_image_description'].index))
        if db_image is None:
            print('Can not find image for the db image description in --data-source-map folder!')
            return

        db_image_keypoints = self._detector.detect(cv2.cvtColor(db_image, cv2.COLOR_BGR2GRAY))
        frame_image_keypoints = self._detector.detect(cv2.cvtColor(match['frame'], cv2.COLOR_BGR2GRAY))

        result_image = cv2.drawMatchesKnn(db_image, db_image_keypoints, match['frame'], frame_image_keypoints,
                                          match['good_matches'], None, flags=2)
        cv2.imshow('best-match', result_image)
        cv2.waitKey(0)
