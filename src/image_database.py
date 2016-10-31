from __future__ import division

import logging
import time
import os

from image_description import ImageDescription


class ImageDatabase(object):
    def __init__(self, options):
        start = time.time()
        self.options = options
        self.root = options.db_path
        self.logger = logging.getLogger(__name__)

        # Configure the feature extractor and feature matcher used by the
        # ImageDescription class.
        ImageDescription.init(options)

        if os.path.isdir(self.root):
            directories = ["{}/{}".format(self.root, identifier) for
                           identifier in os.listdir(self.root)]
            self.items = [ImageDescription.from_directory(directory) for
                          directory in directories]
        else:
            self.items = []

        self.logger.debug("Loaded database in %ss", time.time() - start)

    # Given an image and an audio label for it, this method does feature
    # detection on the image, creates a new ImageDescription object,
    # persists the item to disk and returns the ImageDescription object
    def add(self, image_data, audio_data, description=None):
        # We'll never add more than one image per second, so use a timestamp id.
        dir_name = "{}/{}".format(self.root, time.strftime("%Y%m%dT%H%M%S"))

        if description is None:
            description = ImageDescription.from_image(image_data)
        description.save(dir_name, audio_data, image_data)
        self.items.append(description)

        return description

    # Match the specified image against the database of images. The return value
    # is an array containing zero or more (score, image_desc) tuples.
    def match(self, image_data):
        start = time.time()
        target = ImageDescription.from_image(image_data)

        scores = []
        for _, item in enumerate(self.items):
            score = target.compare_to(item)
            scores.append((score, item))

        scores.sort(key=lambda s: s[0], reverse=True)

        # Loop though the scores until we find one that is bigger than the
        # threshold, or significantly bigger than the best score and then return
        # all the matches above that one.
        retval = []
        best_score = scores[0][0] if len(scores) > 0 else 0
        if best_score >= self.options.matching_score_threshold:
            retval.append(scores[0])
            for score in scores[1:]:
                if score[0] >= self.options.matching_score_threshold and \
                   score[0] >= best_score * self.options.matching_score_ratio:
                    retval.append(score)
                else:
                    break

        self.logger.debug("Matched against %s images in %ss", len(self.items),
                          time.time() - start)
        return retval
