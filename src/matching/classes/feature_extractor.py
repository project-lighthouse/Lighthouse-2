import cv2
import datetime
import errno
import glob
import json
import numpy
import os
import time

from .image_description import ImageDescription


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class FeatureExtractor:
    def __init__(self, verbose):
        self._verbose = verbose

    def extract(self, image_set_path, detector_type, options):
        if detector_type == 'orb':
            # Initialize the ORB descriptor, then detect keypoints and extract local invariant descriptors from the
            # image.
            detector = cv2.ORB_create(nfeatures=options['orb_n_features'])
        elif detector_type == 'akaze':
            detector = cv2.AKAZE_create(descriptor_channels=options['akaze_n_channels'])
        else:
            detector = cv2.xfeatures2d.SURF_create(hessianThreshold=options['surf_threshold'])

        image_descriptions = []

        # loop over the images to find the template in
        for image_set_path in glob.glob(image_set_path + "/**/*.jpg"):
            # File name is an sub_key, file folder is the key.
            key, sub_key = os.path.splitext(image_set_path)[0].split('/')[-2:]

            # Load the image, convert it to grayscale.
            image = cv2.imread(image_set_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if self._verbose:
                print('{} image loaded: {:%H:%M:%S.%f}'.format(image_set_path, datetime.datetime.now()))

            (image_keypoints, image_descriptors) = detector.detectAndCompute(gray_image, None)

            if self._verbose:
                print('{} image\'s features are extracted: {:%H:%M:%S.%f}'.format(image_set_path,
                                                                                  datetime.datetime.now()))

            image_histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            image_histogram = cv2.normalize(image_histogram, image_histogram).flatten()

            if self._verbose:
                print('{} image\'s histogram calculated: {:%H:%M:%S.%f}'.format(image_set_path,
                                                                                datetime.datetime.now()))

            image_descriptions.append(ImageDescription(image_descriptors, image_histogram, key, sub_key))

        if self._verbose:
            print('All images processed ({} images): {:%H:%M:%S.%f}'.format(len(image_descriptions),
                                                                            datetime.datetime.now()))

        return image_descriptions

    def serialize(self, image_descriptions, output_path):
        serialize_time = time.time()

        for image_description in image_descriptions:
            if self._verbose:
                print('Serializing descriptions for {} : {:%H:%M:%S.%f}'.format(image_description.key,
                                                                                datetime.datetime.now()))
            serialized_image_description = {'histogram': dict(dtype=str(image_description.histogram.dtype),
                                                              content=image_description.histogram.tolist()),
                                            'descriptors': dict(dtype=str(image_description.descriptors.dtype),
                                                                content=image_description.descriptors.tolist())}
            output_key_path = '{}/{}'.format(output_path, image_description.key)
            make_dir(output_key_path)

            path = '%s/%s.json' % (output_key_path, image_description.sub_key)
            with open(path, 'w') as outfile:
                json.dump(serialized_image_description, outfile)

        if self._verbose:
            print('All descriptions (%s records) have been serialized in %s seconds' %
                  (len(image_descriptions), time.time() - serialize_time))

    def deserialize(self, input_path):
        image_descriptions = []
        deserialize_time = time.time()

        # loop over the images to find the template in
        for feature_file_path in glob.glob(input_path + '/**/*.json'):
            # File name is an sub_key, file folder is the key.
            key, sub_key = os.path.splitext(feature_file_path)[0].split('/')[-2:]

            if self._verbose:
                print('Deserializing descriptions for %s/%s' % (key, sub_key))

            with open(feature_file_path, 'r') as input_file:
                serialized_image_description = json.load(input_file)

            descriptors = serialized_image_description['descriptors']
            histogram = serialized_image_description['histogram']
            image_descriptions.append(ImageDescription(numpy.array(descriptors['content'], dtype=descriptors['dtype']),
                                                       numpy.array(histogram['content'], dtype=histogram['dtype']),
                                                       key, sub_key))
        if self._verbose:
            print('All descriptions (%s records) have been deserialized in %s seconds.' %
                  (len(image_descriptions), time.time() - deserialize_time))

        return image_descriptions
