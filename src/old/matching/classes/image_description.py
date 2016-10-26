class ImageDescription:
    def __init__(self, descriptors, histogram, key=None, sub_key=None):
        self.descriptors = descriptors
        self.histogram = histogram
        self.key = key
        self.sub_key = sub_key
