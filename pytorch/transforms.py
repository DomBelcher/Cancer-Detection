import numpy as np
import cv2

class Downsize(object):
    '''
        Downsize image
    '''

    def __init__(self, shape=(48, 48), start_shape=(96, 96)):
        self.shape = shape
        self.start_shape = start_shape

        self.start_x = int(np.floor((start_shape[0] - shape[0])/2))
        self.end_x = self.start_x + shape[0]

        self.start_y = int(np.floor((start_shape[1] - shape[1]) / 2))
        self.end_y = self.start_x + shape[1]

    def __call__(self, sample):
        return sample[self.start_x:self.end_x, self.start_y:self.end_y, :]

class RandomRotation(object):
    '''
        Performs random rotation on image
    '''

    def __init__(self, range=(0, 360)):
        self.low = range[0]
        self.high = range[1]

    def __call__(self, sample):
        rows, cols, c = sample.shape
        theta = np.random.randint(self.low, self.high)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
        return cv2.warpAffine(sample,M,(cols,rows))


class Normalise(object):
    def __call__(self, sample):
        return sample/255

class ChannelShift(object):
    def __call__(self, sample):
        return np.moveaxis(sample, -1, 0)

class PytorchToNumpy(object):
    def __call__(self, sample):
        return sample.numpy()

class PillowToNumpy(object):
    def __call__(self, sample):
        return np.array(sample)