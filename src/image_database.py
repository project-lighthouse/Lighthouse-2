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

        self.logger.debug("Image with %s features was added to the database",
                          len(description.features))

        return description

    # Match the specified image against the database of images. The return value
    # is an array containing zero or more (score, image_desc) tuples.
    def match(self, image_data):
        start = time.time()
        target = ImageDescription.from_image(image_data)

        self.logger.debug("Image to find a match for has %s features.",
                          len(target.features))

        scores = [(target.compare_to(item), item) for item in self.items]
        scores.sort(key=lambda s: s[0], reverse=True)

        self.logger.debug("Matched against %s images in %ss", len(self.items),
                          time.time() - start)

        return scores
