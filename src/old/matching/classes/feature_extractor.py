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

    def serialize(self, image_descriptions, output_path):
        serialize_time = time.time()

        for image_description in image_descriptions:
            if self._verbose:
                print('Serializing %s descriptors for %s' % (len(image_description.descriptors), image_description.key))
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
