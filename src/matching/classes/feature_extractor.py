import cv2
import datetime
import glob
import json
import numpy

from .image_description import ImageDescription


class FeatureExtractor:
    def __init__(self, verbose):
        self.verbose = verbose

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
        for image_set_path in glob.glob(image_set_path + "/*.jpg"):
            # Load the image, convert it to grayscale.
            image = cv2.imread(image_set_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if self.verbose:
                print('{} image loaded: {:%H:%M:%S.%f}'.format(image_set_path, datetime.datetime.now()))

            (image_keypoints, image_descriptors) = detector.detectAndCompute(gray_image, None)

            if self.verbose:
                print('{} image\'s features are extracted: {:%H:%M:%S.%f}'.format(image_set_path,
                                                                                  datetime.datetime.now()))

            image_histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            image_histogram = cv2.normalize(image_histogram, image_histogram).flatten()

            if self.verbose:
                print('{} image\'s histogram calculated: {:%H:%M:%S.%f}'.format(image_set_path,
                                                                                datetime.datetime.now()))

            image_descriptions.append(ImageDescription(image_set_path, image_descriptors, image_histogram))

        if self.verbose:
            print('All images processed ({} images): {:%H:%M:%S.%f}'.format(len(image_descriptions),
                                                                            datetime.datetime.now()))

        return image_descriptions

    def serialize(self, image_descriptions, output_path):
        serialized_image_descriptions = []

        for image_description in image_descriptions:
            if self.verbose:
                print('Serializing descriptions for {} : {:%H:%M:%S.%f}'.format(image_description.key,
                                                                                datetime.datetime.now()))
            serialized_image_descriptions.append({'key': image_description.key,
                                                  'histogram': dict(dtype=str(image_description.histogram.dtype),
                                                                    content=image_description.histogram.tolist()),
                                                  'descriptors': dict(dtype=str(image_description.descriptors.dtype),
                                                                      content=image_description.descriptors.tolist())})
        if self.verbose:
            print('All descriptions serialized, writing to file "{}" : {:%H:%M:%S.%f}'.format(output_path,
                                                                                              datetime.datetime.now()))

        with open(output_path, 'w') as outfile:
            json.dump(serialized_image_descriptions, outfile)

    def deserialize(self, input_path):
        with open(input_path, 'r') as input_file:
            serialized_image_descriptions = json.load(input_file)

        if self.verbose:
            print('Serialized data loaded ({} records): {:%H:%M:%S.%f}'.format(
                str(len(serialized_image_descriptions)), datetime.datetime.now()))

        image_descriptions = []
        for serialized_image_description in serialized_image_descriptions:
            if self.verbose:
                print('Deserializing descriptions for {} : {:%H:%M:%S.%f}'.format(serialized_image_description['key'],
                                                                                  datetime.datetime.now()))
            descriptors = serialized_image_description['descriptors']
            histogram = serialized_image_description['histogram']
            image_descriptions.append(
                ImageDescription(serialized_image_description['key'],
                                 numpy.array(descriptors['content'], dtype=descriptors['dtype']),
                                 numpy.array(histogram['content'], dtype=histogram['dtype'])))
        if self.verbose:
            print('All descriptions deserialized: {:%H:%M:%S.%f}'.format(datetime.datetime.now()))

        return image_descriptions
