import cv2
import logging
import numpy
import time
import uuid

from .classes.storage_manager import StorageManager
from .classes.image_description import ImageDescription

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6


class Matcher:
    """Class that is responsible for matching new images against known data set, adding new images to the set etc."""

    def __init__(self, options):
        """Initializes Matcher with provided options."""
        self._options = options
        self._logger = logging.getLogger(__name__)
        self._detector, norm = self._get_detector()
        self._matcher = self._get_matcher(norm)
        self._storage_manager = StorageManager(options['db_path'])
        self._stored_descriptions = None

    def preload_db(self):
        """ Causes StorageManager to preload data into memory"""
        self._stored_descriptions = self._storage_manager.load()

    def match(self, images):
        """Finds the best match for the provided images in the available storage.

        Args:
            images: List of the images to use for matching.

        Returns:
            A tuple that consists of three elements: the first one is the dictionary that describes the best match
            (score, corresponding storage entry, index of the related image and number of "good" matches), the second
            element is a list of image - image description pairs and third one is the input images list.
        """

        best_match = dict(score=0, description=None, image_index=None, good_matches=None)
        ratio_test_coefficient = self._options['matching_ratio_test_k']
        image_descriptions = []

        # Iterate through provided images to find the match for every one and return match with the best score.
        for image_index, image in enumerate(images):
            image_description = self.get_image_description(image)

            image_descriptions.append(image_description)

            # If we couldn't get image description (aka keypoints), let's skip this image.
            if image_description is None:
                continue

            match_frame_start = time.time()
            for idx, stored_description in enumerate(self._stored_descriptions):
                matches = self._matcher.knnMatch(image_description.descriptors, stored_description.descriptors, k=2)

                # Apply ratio test.
                good_matches = []
                for match in matches:
                    if len(match) == 2 and match[0].distance < ratio_test_coefficient * match[1].distance:
                        good_matches.append([match[0]])

                histogram_comparison_result = cv2.compareHist(image_description.histogram, stored_description.histogram,
                                                              cv2.HISTCMP_CORREL)

                matches_count = len(matches)
                if matches_count == 0:
                    score = 0
                else:
                    good_matches_count = len(good_matches)
                    db_image_descriptors_count = len(stored_description.descriptors)

                    score = (1 - numpy.absolute(len(image_description.descriptors) - db_image_descriptors_count) /
                             db_image_descriptors_count) * \
                            (good_matches_count / float(matches_count) * 100) + histogram_comparison_result

                self._logger.debug('Score for %s/%s is %s.' % (stored_description.key, stored_description.sub_key,
                                                               score))

                if score > best_match['score']:
                    best_match['score'] = score
                    best_match['description'] = stored_description
                    best_match['image_index'] = image_index
                    best_match['good_matches'] = good_matches

            self._logger.debug('Image %s is processed in %s seconds.' % (image_index, time.time() - match_frame_start))

        best_match = best_match if best_match['score'] > 0 else None

        return best_match, image_descriptions, images

    def remember_image(self, image, key):
        """Extracts image's feature keypoints and color histogram. Then saves it to the storage under specified key. If
        we can't extract enough keypoints from the image, we return `False` to notify user about the issue.

        Args:
            image: Image to extract features from.
            key: Key to associate image with.

        Returns:
            'True' if image has been successfully added to the storage, otherwise - 'False'.
        """

        # First let's extract image features.
        description = self.get_image_description(image)
        if description is None:
            return False

        # Fill out key and sub_key.
        description.key = key
        description.sub_key = uuid.uuid4()

        self._stored_descriptions.append(description)

        # Persist new entry to the storage.
        self._storage_manager.save_entry(description, image if self._options['db_store_images'] else None)

        return True

    def get_image_description(self, image):
        """Extracts image's feature keypoints and color histogram. If for some reason we can't extract enough keypoints
        we return 'None'.

        Args:
            image: Image to extract features and histogram from.

        Return:
            Instance of ImageDescription if we could extract enough keypoints, otherwise - 'None'.
        """

        histogram_time = time.time()

        # Calculate frame color histogram.
        histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()

        self._logger.debug('Image histogram calculated in %s seconds.' % (time.time() - histogram_time))

        keypoints_time = time.time()

        # Extract all possible keypoints from the frame.
        (keypoints, descriptors) = self._detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)

        keypoints = keypoints if keypoints is not None else []
        descriptors = descriptors if descriptors is not None else []

        self._logger.debug('Image keypoints (%s) and descriptors (%s) have been extracted in %s seconds.' %
                           (len(keypoints), len(descriptors), time.time() - keypoints_time))

        # If we can't extract enough keypoints, that means that something is wrong with the image: either it's too dark
        # or blurry.
        if len(keypoints) < self._options['matching_keypoints_threshold']:
            return None

        return ImageDescription(descriptors, histogram)

    def draw_match(self, match, image):
        """Draws lines between matched keypoints in the available image and target image using provided "match"
        dictionary.

        Args:
            match: Result of image matching returned from `match` method.
            image: Image we used to get the `match` result above.
        """
        stored_image = self._storage_manager.load_entry_image(match['description'])
        if stored_image is None:
            print('Can not find image for the stored image description in --db-path folder!')
            return

        stored_image_keypoints = self._detector.detect(cv2.cvtColor(stored_image, cv2.COLOR_BGR2GRAY))
        image_keypoints = self._detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        result_image = cv2.drawMatchesKnn(stored_image, stored_image_keypoints, image, image_keypoints,
                                          match['good_matches'], None, flags=2)
        cv2.imshow('best-match', result_image)
        cv2.waitKey(0)

    def _get_detector(self):
        """Depending on the matching_detector option this method can return instance of different feature detectors.

        Returns:
            Instance of either ORB detector or AKAZE detector or SURF detector.
        """

        detector_type = self._options['matching_detector']

        if detector_type == 'orb':
            detector = cv2.ORB_create(nfeatures=self._options['matching_orb_n_features'])
            norm = cv2.NORM_HAMMING
        elif detector_type == 'akaze':
            detector = cv2.AKAZE_create(descriptor_channels=self._options['matching_akaze_n_channels'])
            norm = cv2.NORM_HAMMING
        else:
            detector = cv2.xfeatures2d.SURF_create(hessianThreshold=self._options['matching_surf_threshold'])
            norm = cv2.NORM_L2

        return detector, norm

    def _get_matcher(self, norm):
        """Depending on the matching_matcher option this method can return instance of different feature matchers.

        Args:
            norm: Specifies the distance measurement to be used for matching.

        Returns:
            Instance of either BF Matcher or FLANN Based matcher.
        """
        matcher_type = self._options['matching_matcher']

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
