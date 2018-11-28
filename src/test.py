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

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import itertools

import pcl


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


from sklearn.decomposition import PCA
import numpy as np


def denoise(cloud):
    centered = cloud - np.mean(cloud, axis=0)
    pc = PCA(n_components=3)
    fitted = pc.fit_transform(centered)

    # Filter for 90% quantile in 2nd principal direction
    filtered = fitted[fitted[:, 1] < np.quantile(fitted[:, 1], q=0.9)]
    #  filtered = filtered[filtered[:,2] > np.quantile(filtered[:,2], q=0.05) and filtered[:,2] < np.quantile(filtered[:,2], q=0.95)]

    return filtered


def threshold_depth_map(points, kernel_size):
    ''' returns foreground mask'''
    pointcloud = points
    rgb_map = pointcloud_to_rgb_map(pointcloud, target_width=128, target_height=128, scale_factor=1.0,
                                    axis="horizontal")

    depth_map, _, _ = np.dsplit(rgb_map, 3)
    depth_map = depth_map.squeeze()

    depth_normalizer = Normalizer_min_max()
    norm_depth_map = depth_normalizer.forward(depth_map)
    thresholded = otsu_thresholding(norm_depth_map, kernel_size=kernel_size)
    return np.logical_not(thresholded)


def otsu_thresholding(img, kernel_size=5):
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    ret3, th3 = cv2.threshold(blur.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


class SegmentationData():
    def __init__(self, root_dir=r'/home/sandro/Dokumente/WHo/denseposeoutputs/'):
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

    seg_image = seg_image
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


def maybe_roll_seg_image(rgb_image, seg_image):
    rgb_shape = rgb_image.shape[:2]
    seg_shape = seg_image.shape
    if not rgb_shape[0] == seg_shape[0]:
        seg_image = np.rollaxis(seg_image, 1, 0)
        seg_shape = seg_image.shape
    assert seg_shape == rgb_shape
    return seg_image


def segment_foreground(pcd_file, jpg_file, seg_file, save_ply=True):
    cloud = PyntCloud.from_file(pcd_file)  # :PyntCloud
    rgb_image = cv2.imread(jpg_file, -1).astype(np.uint8)[::-1, :, :]
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    seg_image = cv2.imread(seg_file, -1).astype(np.uint8)
    seg_image = maybe_roll_seg_image(rgb_image, seg_image)
    seg_image = seg_image[::-1, :]

    points = cloud.points.values[:, :3]
    seg_vals = transfer_segmentation(seg_image, points)
    color_vals = transfer_image(rgb_image, points)
    color_seg_vals = colorize_seg_vals(seg_vals)

    obj_ids, counts = np.unique(seg_image, return_counts=True)
    counts[0] = 0  # foreground - dont care

    largest_obj = np.argmax(counts)
    fg_array = points[seg_vals == obj_ids[largest_obj]]

    output_file_name = os.path.basename(pcd_file)
    output_file_name = os.path.splitext(output_file_name)[0]

    if save_ply:
        print(output_file_name)
        write_color_ply(output_file_name + '_color_seg.ply', points, color_seg_vals)
        write_color_ply(output_file_name + '_rgb.ply', points, color_vals)
        write_color_ply(output_file_name + '_fg.ply', fg_array, color_seg_vals[seg_vals == 1])
        np.savetxt(output_file_name + '_fg.txt', fg_array)
    return fg_array


def get_bounding_box_height(pcd_path_or_array, from_array=False):
    if from_array:
        p = pcl.from_array(pcd_path_or_array)
    else:
        p = pcl.load(pcd_path_or_array)

    feature_extractor = pcl.MomentOfInertiaEstimation()
    feature_extractor.set_InputCloud(p)
    feature_extractor.compute()

    [min_point_AABB, max_point_AABB] = feature_extractor.get_AABB()
    eccentricity = feature_extractor.get_Eccentricity()
    [major_value, middle_value, minor_value] = feature_extractor.get_EigenValues()
    [major_vector, middle_vector, minor_vector] = feature_extractor.get_EigenVectors()
    mass_center = feature_extractor.get_MassCenter()
    moment_of_inertia = feature_extractor.get_MomentOfInertia()
    [min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB] = feature_extractor.get_OBB()

    bounding_box_coordinates = max_point_OBB - min_point_OBB

    bounding_box_height = bounding_box_coordinates[0][0]

    return bounding_box_height


def get_bbox_height(pcd_path_or_array, from_array=False):
    if from_array:
        p = pcl.PointCloud()
        p.from_array(pcd_path_or_array)
    else:
        p = pcl.load(pcd_path_or_array)
    feature_extractor = pcl.MomentOfInertiaEstimation()
    feature_extractor.set_InputCloud(p)
    feature_extractor.compute()

    [min_point_AABB, max_point_AABB] = feature_extractor.get_AABB()
    eccentricity = feature_extractor.get_Eccentricity()
    [major_value, middle_value, minor_value] = feature_extractor.get_EigenValues()
    [major_vector, middle_vector, minor_vector] = feature_extractor.get_EigenVectors()
    mass_center = feature_extractor.get_MassCenter()
    moment_of_inertia = feature_extractor.get_MomentOfInertia()
    [min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB] = feature_extractor.get_OBB()

    # calculate the difference of min and max
    bbox_difference = max_point_OBB - min_point_OBB
    bbox_height = bbox_difference[0][0]

    return min_point_OBB, max_point_OBB, bbox_height, rotational_matrix_OBB, position_OBB


#     return min_point_AABB, max_point_AABB, bbox_height
#     return min_point_OBB_inv, max_point_OBB_inv, bbox_height


def get_points_OBB(min_point_OBB, max_point_OBB, rotational_matrix_OBB, position_OBB):
    p1 = (min_point_OBB[0][0], min_point_OBB[0][1], min_point_OBB[0][2])
    p2 = (min_point_OBB[0][0], min_point_OBB[0][1], max_point_OBB[0][2])
    p3 = (max_point_OBB[0][0], min_point_OBB[0][1], max_point_OBB[0][2])
    p4 = (max_point_OBB[0][0], min_point_OBB[0][1], min_point_OBB[0][2])
    p5 = (min_point_OBB[0][0], max_point_OBB[0][1], min_point_OBB[0][2])
    p6 = (min_point_OBB[0][0], max_point_OBB[0][1], max_point_OBB[0][2])
    p7 = (max_point_OBB[0][0], max_point_OBB[0][1], max_point_OBB[0][2])
    p8 = (max_point_OBB[0][0], max_point_OBB[0][1], min_point_OBB[0][2])

    p1 = rotational_matrix_OBB @ p1 + position_OBB
    p2 = rotational_matrix_OBB @ p2 + position_OBB
    p3 = rotational_matrix_OBB @ p3 + position_OBB
    p4 = rotational_matrix_OBB @ p4 + position_OBB
    p5 = rotational_matrix_OBB @ p5 + position_OBB
    p6 = rotational_matrix_OBB @ p6 + position_OBB
    p7 = rotational_matrix_OBB @ p7 + position_OBB
    p8 = rotational_matrix_OBB @ p8 + position_OBB

    points_OBB = np.asarray([p1, p2, p3, p4, p5, p6, p7, p8])
    points_OBB = points_OBB.reshape(8, 3)

    return points_OBB


def render_pointcloud_with_bbox(pointcloud, bbox_points, save_path, title=None, col=None, cmap="gray"):
    """
    Renders a point-cloud.
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(bbox_points[:, 0], bbox_points[:, 1], bbox_points[:, 2], c='red', s=0.5, cmap=cmap, alpha=0.5)
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c=col, s=0.5, cmap=cmap, alpha=0.5)

    edge = [
        [0, 1], [0, 3], [0, 4],
        [7, 3],
        [1, 5], [1, 2],
        [2, 3], [2, 6],
        [4, 5], [4, 7],
        [5, 6], [6, 7]
    ]

    for e in edge:
        ax.plot(bbox_points[e, 0], bbox_points[e, 1], bbox_points[e, 2], c='red' if e == [5, 6] else 'black',
                lw=4 if e == [5, 6] else 2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    fig.savefig(save_path)

    return fig


def get_bbox_and_height(segmented_file_or_array, save_path, from_array=False):
    if not from_array:
        point_cloud = load_pcd_as_ndarray(segmented_file_or_array)
    else:
        point_cloud = segmented_file_or_array
    min_OBB, max_OBB, height, rot_mat_OBB, position_OBB = get_bbox_height(point_cloud, True)

    points_bbox = get_points_OBB(min_OBB, max_OBB, rot_mat_OBB, position_OBB)
    fig = render_pointcloud_with_bbox(point_cloud, points_bbox, save_path)
    return points_bbox, height