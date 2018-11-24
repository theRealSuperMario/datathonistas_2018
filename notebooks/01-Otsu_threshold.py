import cv2
import numpy as np

from pylab import *
import sys
sys.path.insert(0, '..')

from src.paths import PNG_FILES, PCD_FILES
from src.cgmcore.utils import load_pcd_as_ndarray, pointcloud_to_rgb_map2, show_rgb_map

png_file = PNG_FILES[0]
pcd_file = PCD_FILES[0]

pc_points = load_pcd_as_ndarray(pcd_file)

rgb_map, point_transform_func = pointcloud_to_rgb_map2(pc_points, target_width=128, target_height=128, scale_factor=1.0, axis="horizontal")
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
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(k_size, k_size))
m = (np.logical_not(thresholded) * depth_map) > 0
axes[3].imshow(m)
eroded = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
axes[4].imshow(eroded)
plt.show()


fg_estimate = eroded * depth_map

def filter_within_quantiles(lower_q, upper_q, img):
    lower_th = np.quantile(img, lower_q)
    upper_th = np.quantile(img, upper_q)
    mask = np.logical_and(img < upper_th, img > lower_th)
    return lower_th, upper_th, mask

_, _, new_fg_mask = filter_within_quantiles(0.1, 0.9, fg_estimate)

axes[5].imshow(new_fg_mask)


new_points, c_mask = point_transform_func(pc_points)
new_points = new_points[:, :3] # remove certainty

valid_points = np.zeros_like(new_points)[:, 0].squeeze()
for i in np.arange(valid_points.shape[0]):
    if c_mask[i]:
        x, y, z = new_points[i, :]
        if new_fg_mask[int(x), int(y)]:
            valid_points[i] = 1

final_mask = np.logical_and(c_mask, valid_points == 1)


color = np.array([0, 255, 255])

import pyntcloud.io


pyntcloud.io.write_ply('input_points.ply', pc_points)
pyntcloud.io.write_ply('seg_points.ply', new_points[final_mask])






# h, w, c = rgb_map.shape
# cy, cx = int(h/2), int(w/2)
# seed_depth = depth_map[cy, cx, ...]
#
# center_point = np.array([cy, cx])
# half_rect_size = 10
# p1 = center_point - np.array([half_rect_size, half_rect_size])
# p2 = center_point + np.array([half_rect_size, half_rect_size])
#
# seed_mask = np.zeros_like(depth_map).squeeze()
# seed_mask = cv2.rectangle(seed_mask, tuple(p1), tuple(p2), 1, cv2.FILLED)
#
# fg = depth_map * seed_mask
#
# ## refine foreground
#
# def median_thresholding(x):
#     median = np.median(x[x>0.0])
#     return np.logical_and(x < median, x > 0.0)
#
# median_mask = median_thresholding(fg).astype(np.int)
#
# seed_mask = median_mask
#
#
# class Normalizer_min_max():
#     min = None
#     max = None
#     def __init__(self):
#         pass
#
#     def forward(self, image):
#         if self.min is None:
#             self.min = image.min()
#             self.max = image.max()
#         _range = self.max - self.min
#         return (image - self.min) / _range * 255
#
#     def reset(self):
#         self.min = None
#         self.ma = None
#
#     def backward(self, image):
#         _range = self.max - self.min
#         return image * _range / 255 + self.min
#
#
# depth_normalizer = Normalizer_min_max()
# norm_depth_map = depth_normalizer.forward(depth_map)
#
#
# hist, bin_edges = np.histogram(norm_depth_map, bins=64)
#
