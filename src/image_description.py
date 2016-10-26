from __future__ import division
import numpy as np
import cv2
import os

# This is how we extract features from an image
feature_extractor = cv2.ORB_create()

# And this is how we look for feature sets that match
feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

class ImageDescription:
    # Private constructor. Use one of the factory functions below
    def __init__(self, dirname, features):
        self.dirname = dirname
        self.features = features

    # Factory function that returns an ImageDescription read from the
    # specified directory
    @staticmethod
    def fromDirectory(dirname):
        datafile = "{}/{}".format(dirname, "data.npz")
        with np.load(datafile) as data:
            return ImageDescription(dirname, (data['orb'], data['histogram']))

    # Factory function that returns an ImageDescription created from the
    # specified image data. The returned object does not have a dirname
    # or any associated audio data until it is saved with the write() method
    @staticmethod
    def fromImage(imagedata):
        # Calculate color histogram.
        histogram = cv2.calcHist([imagedata], [0, 1, 2], None,
                                 [8, 8, 8], [0, 256, 0, 256, 0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()

        # Extract all possible keypoints from the frame.
        grayscale = cv2.cvtColor(imagedata, cv2.COLOR_BGR2GRAY)
        (keypoints, features) = feature_extractor.detectAndCompute(grayscale,
                                                                   None)

        return ImageDescription(None, (features, histogram))

    # This method saves the item description to the specified directory.
    # If the image data and audio data are specified, they are also saved
    def save(self, dirname, audiodata=[], imagedata=[]):
        self.dirname = dirname
        os.makedirs(dirname)

        datafile = "{}/{}".format(dirname, "data.npz")
        np.savez(datafile, orb=self.features[0], histogram=self.features[1])

        if len(imagedata):
            imagefile = "{}/{}".format(dirname, "image.jpg")
            cv2.imwrite(imagefile, imagedata)

        if len(audiodata):
            audiofile = "{}/{}".format(dirname, "audio.raw")
            with open(audiofile, "wb") as f:
                f.write(audiodata)

    def audio_filename(self):
        if self.dirname == None:
            return None
        else:
            return "{}/audio.raw".format(self.dirname)

    def image_filename(self):
        if self.dirname == None:
            return None
        else:
            return "{}/image.jpg".format(self.dirname)

    # Compare this item description to another and return a score
    # indicating how well they match. Lower values indicate better matches
    def compare_to(self, other):
        # Compare the ORB features of the two images and compute
        # the average distance for the matches that are foudn
        matches = feature_matcher.match(self.features[0], other.features[0])
        distances = [m.distance for m in matches]
        avgdist = sum(distances) / float(len(distances))

        # compute the histogram correlation and invert it so that
        # better correlations give smaller values
        histcmp = 1 - cv2.compareHist(self.features[1],
                                      other.features[1],
                                      cv2.HISTCMP_CORREL)

        # Use the product of these two numbers to compute a score.
        # Lower numbers mean a better match
        return avgdist * histcmp
