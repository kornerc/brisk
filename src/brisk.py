
import pybrisk

class Brisk:
    def __init__(self):
        self.descriptor_extractor = pybrisk.create()

    def __del__(self):
        pybrisk.destroy(self.descriptor_extractor)

    def detect_and_extract(self, img, thresh=60, octaves=4):
        return pybrisk.detect_and_extract(self.descriptor_extractor,
                img, thresh, octaves)

    def detect_keypoints(self, img, thresh=60, octaves=4):
        return pybrisk.detect_keypoints(img, thresh, octaves)
