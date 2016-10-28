from __future__ import division
from image_description import ImageDescription
import logging
import time
import os

class ImageDatabase:
    def __init__(self, options):
        start = time.time()
        self.options = options
        self.root = options.db_path
        self.logger = logging.getLogger(__name__)


        # Configure the feature extractor and feature matcher
        # used by the ImageDescription class
        ImageDescription.init(options)

        if os.path.isdir(self.root):
            self.items = [ImageDescription.fromDirectory(dir) for dir in
                          ["{}/{}".format(self.root, id) for id in
                           os.listdir(self.root)]]
        else:
            self.items = []

        self.logger.debug("Loaded database in {}s".format(time.time()-start))

    # Given an image and an audio label for it, this method does feature
    # detection on the image, creates a new ImageDescription object,
    # persists the item to disk and returns the ImageDescription object
    def add(self, imagedata, audiodata, description=None):
        if description == None:
            description = ImageDescription.fromImage(imagedata)

        # We'll never add more than one image per second, so use a timestamp id
        id = time.strftime("%Y%m%dT%H%M%S")
        dirname = "{}/{}".format(self.root, id)

        description.save(dirname, audiodata, imagedata)

        self.items.append(description)
        return description

    # Match the specified image against the database of images.
    # The return value is an array containing zero or more
    # (score,image_desc) tuples
    def match(self, imagedata):
        start = time.time()
        target = ImageDescription.fromImage(imagedata)
        scores = []
        for i,item in enumerate(self.items):
            score = target.compare_to(item)
            scores.append((score, item))

        scores.sort(key=lambda s: s[0], reverse=True)

        # Loop though the scores until we find one that is lower than
        # the threshold, or significantly lower than the best score
        # and then return all the matches above that one
        retval = []
        if scores[0][0] >= self.options.matching_score_threshold:
            retval.append(scores[0])
            for i in range(1, len(scores)):
                if scores[i][0] >= self.options.matching_score_threshold and \
                   scores[i][0] >= scores[0][0] * self.options.matching_score_ratio:
                    retval.append(scores[i])
                else:
                    break

        self.logger.debug("Matched against {} images in {}s"
                          .format(len(self.items), time.time()-start))
        return retval

