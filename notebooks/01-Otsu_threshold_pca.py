import cv2
import numpy as np

from pylab import *
import sys
sys.path.insert(0, '..')

from src.paths import PNG_FILES, PCD_FILES
from src.cgmcore.utils import load_pcd_as_ndarray, pointcloud_to_rgb_map, show_rgb_map

png_file = PNG_FILES[0]
pcd_file = PCD_FILES[0]



pointcloud = load_pcd_as_ndarray(pcd_file)

rgb_map = pointcloud_to_rgb_map(pointcloud, target_width=128, target_height=128, scale_factor=1.0, axis="horizontal")
# show_rgb_map(rgb_map)

# rgb_map = pointcloud_to_rgb_map(pointcloud, target_width=128, target_height=128, scale_factor=1.0, axis="vertical")
# show_rgb_map(rgb_map)

# rgb_map = cv2.imread(png_file, -1)

depth_map, density_map, intensity_map = np.dsplit(rgb_map, 3)
depth_map = depth_map.squeeze()
density_map = density_map.squeeze()
intensity_map = intensity_map.squeeze()


class Normalizer_min_max():
    min = None
    max = None
    def __init__(self):
        pass

    def forward(self, image):
        if self.min is None:
            self.min = image.min()
            self.max = image.max()
        _range = self.max - self.min
        return (image - self.min) / _range * 255

    def reset(self):
        self.min = None
        self.ma = None

    def backward(self, image):
        _range = self.max - self.min
        return image * _range / 255 + self.min

depth_normalizer = Normalizer_min_max()
norm_depth_map = depth_normalizer.forward(depth_map)


def otsu_thresholding(img):
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

thresholded = otsu_thresholding(depth_map)


fig, axes = plt.subplots(1, 6)
axes[0].imshow(depth_map)
axes[1].imshow(thresholded)
axes[2].imshow(np.logical_not(thresholded) * depth_map)
k_size = 5
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_size, k_size))
m = (np.logical_not(thresholded) * depth_map) > 0
axes[3].imshow(m)
eroded = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

plt.show()

from sklearn import decomposition



decomposition

