import cv2
import numpy as np

from pylab import *
import sys
sys.path.insert(0, '..')

from src.paths import PNG_FILES, PCD_FILES
from src.cgmcore.utils import load_pcd_as_ndarray, pointcloud_to_rgb_map, show_rgb_map


class Normalizer_min_max():
    min = None
    max = None

    def __init__(self):
        '''
        handy class to normalize an image to 0, 255 forward and backward
        '''
        pass

    def forward(self, image):
        if self.min is None:
            self.min = image.min()
            self.max = image.max()
        _range = self.max - self.min
        return (image - self.min) / _range * 255

    def reset(self):
        '''
        resets transformation parameters
        can be used for chaining:
        normalizer.reset().forward
        '''
        self.min = None
        self.max = None
        return self

    def backward(self, image):
        _range = self.max - self.min
        return image * _range / 255 + self.min


def threshold_depth_map(points, kernel_size):
    ''' returns foreground mask'''
    pointcloud = points
    rgb_map = pointcloud_to_rgb_map(pointcloud, target_width=128, target_height=128, scale_factor=1.0, axis="horizontal")

    depth_map, _, _ = np.dsplit(rgb_map, 3)
    depth_map = depth_map.squeeze()

    depth_normalizer = Normalizer_min_max()
    norm_depth_map = depth_normalizer.forward(depth_map)
    thresholded = otsu_thresholding(norm_depth_map, kernel_size=kernel_size)
    return np.logical_not(thresholded)


def otsu_thresholding(img, kernel_size = 5):
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    ret3, th3 = cv2.threshold(blur.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3