import cv2
import numpy
import numpy as np
import pandas as pd

from pylab import *
import sys

from pyntcloud import PyntCloud
from pyntcloud.io import write_ply

sys.path.insert(0, '..')
from src.tango import get_extrinsic_matrix, get_intrinsic_matrix, get_k
from src.paths import PNG_FILES, PCD_FILES
from src.cgmcore.utils import load_pcd_as_ndarray, pointcloud_to_rgb_map, show_rgb_map
import os
from matplotlib import pyplot as plt

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


class SegmentationData():
    def __init__(self, root_dir = r'/home/sandro/Dokumente/WHo/denseposeoutputs/'):
        self.root_dir = root_dir

    def get_uv(self, mh_num, time_step, some_number, idx):
        path = 'rgb_MH_WHH_{:04d}_{}_{}_{}_IUV.png'.format(mh_num, time_step, some_number, idx)
        return os.path.join(self.root_dir, path)

    def get_inds(self, mh_num, time_step, some_number, idx):
        path = 'rgb_MH_WHH_{:04d}_{}_{}_{}_INDS.png'.format(mh_num, time_step, some_number, idx)
        return os.path.join(self.root_dir, path)


def write_color_ply(fname, points, color_vals):
    df = pd.DataFrame(columns=['x', 'y', 'z', 'red', 'green', 'blue'])
    df['x'] = points[:, 0]
    df['y'] = points[:, 1]
    df['z'] = points[:, 2]
    df['red'] = color_vals[:, 0].astype(np.uint8)
    df['green'] = color_vals[:, 1].astype(np.uint8)
    df['blue'] = color_vals[:, 2].astype(np.uint8)
    new_pc = PyntCloud(df)
    write_ply(fname, new_pc.points, as_text=True)

def transfer_segmentation(seg_image, pc_points):
    hh, ww = seg_image.shape
    ext_d = get_extrinsic_matrix(4)
    r_vec = ext_d[:3, :3]
    t_vec = -ext_d[:3, 3]
    intrinsic = get_intrinsic_matrix()

    k1, k2, k3 = get_k()
    im_coords, _ = cv2.projectPoints(pc_points, r_vec, t_vec, intrinsic[:3, :3], np.array([k1, k2, 0, 0]))

    n_points = pc_points.shape[0]
    seg_vals = np.zeros(shape=(n_points), dtype=seg_image.dtype)

    for i, t in enumerate(im_coords):
        x, y = t.squeeze()
        x = int(np.round(x))
        y = int(np.round(y))
        if x >= 0 and x < ww and y >= 0 and y < hh:
            seg_vals[i] = seg_image[y, x]
    return seg_vals


def transfer_image(image, pc_points):
    hh, ww, _ = image.shape
    ext_d = get_extrinsic_matrix(4)
    r_vec = ext_d[:3, :3]
    t_vec = -ext_d[:3, 3]
    intrinsic = get_intrinsic_matrix()

    k1, k2, k3 = get_k()
    im_coords, _ = cv2.projectPoints(pc_points, r_vec, t_vec, intrinsic[:3, :3], np.array([k1, k2, 0, 0]))

    n_points = pc_points.shape[0]
    color_vals = np.zeros_like(pc_points)

    for i, t in enumerate(im_coords):
        x, y = t.squeeze()
        x = int(np.round(x))
        y = int(np.round(y))
        if x >= 0 and x < ww and y >= 0 and y < hh:
            color_vals[i, :] = image[y, x]
    return color_vals


def colorize_seg_vals(seg_vals):
    '''
    given (N, 1) or (N) array of segmentation ids, returns a color representation for them
    :param seg_vals:
    :return:
    '''
    seg_vals = seg_vals.squeeze()
    nums = np.linspace(0, 1, seg_vals.max() + 1)
    np.random.shuffle(nums)
    color_lookup = plt.cm.coolwarm(nums, bytes=True)

    out_vals = np.array([color_lookup[s] for s in seg_vals])
    return out_vals


def segment_foreground(pcd_file, jpg_file, seg_file, save_ply=True):
    cloud = PyntCloud.from_file(pcd_file)  # :PyntCloud
    rgb_image = cv2.imread(jpg_file, -1).astype(np.uint8)[::-1, :, :]
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    seg_image = cv2.imread(seg_file, -1).astype(np.uint8)
    seg_image = np.rollaxis(seg_image, 1, 0)
    points = cloud.points.values[:, :3]

    seg_vals = transfer_segmentation(seg_image, points)
    color_vals = transfer_image(rgb_image, points)


    color_seg_vals = colorize_seg_vals(seg_vals)
    fg_array = points[seg_vals == 1]

    output_file_name = os.path.basename(pcd_file)
    output_file_name = os.path.splitext(pcd_file)[0]

    if save_ply:
        write_color_ply(output_file_name + '_color_seg.ply', points, color_seg_vals)
        write_color_ply(output_file_name + '_rgb.ply', points, color_vals)
        np.savetxt(output_file_name + '_fg.txt')
    return fg_array