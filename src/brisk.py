
import pybrisk

class Brisk:
    def __init__(self, thresh=60, octaves=4):
        self.thresh = thresh
        self.octaves = octaves
        self.descriptor_extractor = pybrisk.create()

    def __del__(self):
        pybrisk.destroy(self.descriptor_extractor)

    def detect(self, img):
        return pybrisk.detect(self.descriptor_extractor,
                img, self.thresh, self.octaves)

    def compute(self, img, keypoints):
        return pybrisk.compute(self.descriptor_extractor,
                img, keypoints)
