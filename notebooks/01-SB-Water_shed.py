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
show_rgb_map(rgb_map)

# rgb_map = pointcloud_to_rgb_map(pointcloud, target_width=128, target_height=128, scale_factor=1.0, axis="vertical")
# show_rgb_map(rgb_map)

# rgb_map = cv2.imread(png_file, -1)

depth_map, density_map, intensity_map = np.dsplit(rgb_map, 3)
depth_map = depth_map.squeeze()
density_map = density_map.squeeze()
intensity_map = intensity_map.squeeze()

h, w, c = rgb_map.shape
cy, cx = int(h/2), int(w/2)
seed_depth = depth_map[cy, cx, ...]

center_point = np.array([cy, cx])
half_rect_size = 10
p1 = center_point - np.array([half_rect_size, half_rect_size])
p2 = center_point + np.array([half_rect_size, half_rect_size])

seed_mask = np.zeros_like(depth_map).squeeze()
seed_mask = cv2.rectangle(seed_mask, tuple(p1), tuple(p2), 1, cv2.FILLED)

fg = depth_map * seed_mask

## refine foreground

def median_thresholding(x):
    median = np.median(x[x>0.0])
    return np.logical_and(x < median, x > 0.0)

median_mask = median_thresholding(fg).astype(np.int)

seed_mask = median_mask


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


hist, bin_edges = np.histogram(norm_depth_map, bins=64)

