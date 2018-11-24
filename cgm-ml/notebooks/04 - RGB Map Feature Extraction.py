
# This makes sure that you got access to the cgmcore-module
import sys
sys.path.insert(0, "..")

# Retrieving the latest ETL-data-path.
from cgmcore.etldatagenerator import get_dataset_path
# dataset_path = get_dataset_path("../../data/etl")
# print("Using daataset-path {}.".format(dataset_path))
#
# # Getting all PCDs.
import glob2 as glob
import os
# all_pcd_paths = glob.glob(os.path.join(dataset_path, "**/*.pcd"))
# print("Found {} PCD-files.".format(len(all_pcd_paths)))
#
# # Randomly selecting one PCD-path.
import random
# random_pcd_path = random.choice(all_pcd_paths)
# print("Using random PCD-path {}.".format(random_pcd_path))

# Load the pointcloud from the PCD-path.
from cgmcore.utils import load_pcd_as_ndarray, pointcloud_to_rgb_map
from pylab import *
path = r'/home/sandro/Dokumente/WHo/MH_WHH_0008/1537167614128/pcd/pc_MH_WHH_0008_1537167343405_104_000.pcd'
pointcloud = load_pcd_as_ndarray(path)
print("Loaded pointcloud with shape {}.".format(pointcloud.shape))



rgb_map = pointcloud_to_rgb_map(pointcloud, target_width=128, target_height=128, scale_factor=1.0, axis="horizontal")
print("Shape of RGB-map is {}.".format(rgb_map.shape))

### to

from cgmcore.utils import show_rgb_map

show_rgb_map(rgb_map)

import numpy as np
depth_map, density_map, intensity_map = np.dsplit(rgb_map, 3)

h, w, c = rgb_map.shape
cy, cx = int(h/2), int(w/2)

seed_depth = depth_map[cy, cx, ...]

def get_neighbors(points, seed_depth, tolerance):
    neighbors


pc_coords = pointcloud[:, :3]
pc_mean

mean, eigvectors = cv2.PCACompute(pointcloud, pointcloud.mean())

np.linalg.svd()






# In[ ]:




