import cv2
import errno
import glob
import json
import logging
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


class StorageManager:
    """Class that is responsible for loading and (de-)serialization of image feature descriptors, histograms etc."""

    def __init__(self, work_dir):
        """Initializes StorageManager with `work-dir`."""

        self._work_dir = work_dir
        self._logger = logging.getLogger(__name__)

    def load(self):
        """Loads and returns entire storage

        Returns:
            A list of ImageDescription instances loaded from the storage.
        """

        entries = []
        load_time = time.time()

        # Loop over all JSON files in the working directory and try to deserialize all of them.
        for entry_path in glob.glob('%s/**/*.json' % self._work_dir):
            # File name is an sub_key, file folder is the key.
            key, sub_key = os.path.splitext(entry_path)[0].split('/')[-2:]

            self._logger.debug('Loading %s/%s' % (key, sub_key))

            # TODO: Process all possible exceptions we can get here.
            with open(entry_path, 'r') as entry_file:
                serialized_entry = json.load(entry_file)

            descriptors = serialized_entry['descriptors']
            histogram = serialized_entry['histogram']
            entries.append(ImageDescription(numpy.array(descriptors['content'], dtype=descriptors['dtype']),
                                            numpy.array(histogram['content'], dtype=histogram['dtype']), key, sub_key))

        self._logger.debug('All database entries (%s) have been loaded and deserialized in %s seconds.' %
                           (len(entries), time.time() - load_time))

        return entries

    def load_entry_image(self, entry):
        """Loads image for the specified entry if exists

        Args:
            entry: Stored entry we'd like to load image for.

        Returns:
            OpenCV image object if image exists, otherwise - None
        """
        return cv2.imread('%s/%s/%s.jpg' % (self._work_dir, entry.key, entry.sub_key))

    def save_entry(self, entry, image=None):
        """Saves single entry to the storage

        Args:
            entry: ImageDescription instance to be saved to storage.
            image: Optional image for the image description. If provided - will be saved as well.
        """

        save_time = time.time()

        self._logger.debug('Saving %s/%s with %s descriptors.' % (entry.key, entry.sub_key,
                                                                  len(entry.descriptors)))
        serialized_entry = {'histogram': dict(dtype=str(entry.histogram.dtype), content=entry.histogram.tolist()),
                            'descriptors': dict(dtype=str(entry.descriptors.dtype), content=entry.descriptors.tolist())}

        # Let's create directory for the 'key' if it doesn't exist yet.
        entry_folder_path = '%s/%s' % (self._work_dir, entry.key)
        make_dir(entry_folder_path)

        with open('%s/%s.json' % (entry_folder_path, entry.sub_key), 'w') as entry_file:
            json.dump(serialized_entry, entry_file)

        self._logger.debug('Entry %s/%s has been serialized and saved in %s seconds' % (entry.key, entry.sub_key,
                                                                                        time.time() - save_time))

        # If image itself is provided, let's save it to the storage as well.
        if image is not None:
            self._logger.debug('Saving image for entry %s/%s' % (entry.key, entry.sub_key))
            cv2.imwrite('%s/%s.jpg' % (entry_folder_path, entry.sub_key), image)

