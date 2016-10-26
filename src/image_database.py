from __future__ import division
from image_description import ImageDescription
import time
import os

class ImageDatabase:
    def __init__(self, rootdir):
        self.root = rootdir
        if os.path.isdir(self.root):
            self.items = [ImageDescription.fromDirectory(dir) for dir in
                          ["{}/{}".format(rootdir, id) for id in
                           os.listdir(self.root)]]
        else:
            self.items = []

    # Given an image and an audio label for it, this method does feature
    # detection on the image, creates a new ImageDescription object,
    # persists the item to disk and returns the ImageDescription object
    def add(self, imagedata, audiodata):
        item = ImageDescription.fromImage(imagedata)

        # We'll never add more than one image per second, so use a timestamp id
        id = time.strftime("%Y%m%dT%H%M%S")
        dirname = "{}/{}".format(self.root, id)

        item.save(dirname, audiodata, imagedata)

        self.items.append(item)
        return item

    def match(self, imagedata):
        target = ImageDescription.fromImage(imagedata)
        bestScore = 1000000
        bestMatch = None
        for i in range(0, len(self.items)):
            item = self.items[i]
            score = item.compare_to(target)
            print(score, item.dirname)
            if (score < bestScore):
                bestScore = score
                bestMatch = item
        return bestMatch
