import cv2
import numpy as np

from pylab import *
import sys
sys.path.insert(0, '..')
import pandas as pd
from src.paths import PNG_FILES, PCD_FILES
from src.cgmcore.utils import load_pcd_as_ndarray, pointcloud_to_rgb_map2, show_rgb_map
import cv2

from pylab import *
from pyntcloud import PyntCloud
from pyntcloud.io import write_ply

pcd_file = r'/home/sandro/Dokumente/WHo/PCD/1536913514153/pcd/pc_MH_WHH_0001_1536913319075_104_000.pcd'
# jpg_file = r'/home/sandro/Dokumente/WHo/PCD/1536911714062/jpg/rgb_MH_WHH_0014_1536911466814_104_5328.379994052.jpg'
jpg_file = r'/home/sandro/Dokumente/WHo/PCD/1536913514153/jpg/rgb_MH_WHH_0001_1536913319075_104_7137.838129759.jpg'

calib_file = r'calibration.xml'


cloud = PyntCloud.from_file(pcd_file)  # :PyntCloud

jpg = cv2.imread(jpg_file, -1)
jpg = jpg

hh, ww, _ = jpg.shape

import xmltodict

with open(calib_file) as fd:
    calib = xmltodict.parse(fd.read())

points = cloud.points.values[:, :3]


def get_intrinsic_matrix(calib):
    arr = calib['rig']['camera'][1]['camera_model']['params']
    fu, fv, u0, v0, k1, k2, k3 = np.fromstring(arr.replace('[', '').replace(']', ''), sep=';')
    _gamma = 1
    intrinsic = np.array(
        [[fu, _gamma, u0, 0],
         [0, fv, v0, 0],
         [0, 0, 1, 0]])
    return intrinsic

def get_k(calib):
    arr = calib['rig']['camera'][1]['camera_model']['params']
    fu, fv, u0, v0, k1, k2, k3 = np.fromstring(arr.replace('[', '').replace(']', ''), sep=';')
    _gamma = 1
    intrinsic = np.array(
        [[fu, 0, u0, 0],
         [0, fv, v0, 0],
         [0, 0, 1, 0]])
    return k1, k2, k3

def get_extrinsic_matrix(calib, idx=1):
    arr = calib['rig']['extrinsic_calibration'][idx]['A_T_B']
    arr = arr.split(';')
    arr = [x.replace('[', '').replace(']', '') for x in arr]
    mat = np.array([np.fromstring(x, sep=',') for x in arr])
    mat[:3, :3] = mat[:3, :3].T # maybe transpose?
    mat = np.vstack([mat, np.array([0, 0, 0, 1])])
    return mat

intrinsic = get_intrinsic_matrix(calib)
ext_d = get_extrinsic_matrix(calib, 4)
ext_rgb = get_extrinsic_matrix(calib, 3)

# diff = ext_rgb @ ext_d
diff = ext_rgb @ np.linalg.inv(ext_d)
# diff = np.linalg.inv(ext_rgb) @ ext_d # best so far
# diff = np.linalg.inv(ext_rgb) @ np.linalg.inv(ext_d)

# diff = ext_d @ ext_rgb
# diff = ext_d @ np.linalg.inv(ext_rgb)
# diff = np.linalg.inv(ext_d) @ ext_rgb
# diff = np.linalg.inv(ext_d) @ np.linalg.inv(ext_rgb)


Hpoints = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
Hpoints = np.transpose(diff @ Hpoints.T)

im_coords = np.transpose(intrinsic @ Hpoints.T)


color_vals = np.zeros_like(points)

for i, t in enumerate(im_coords):
    x, y, _ = t
    x = int(np.round(x))
    y = int(np.round(y))
    if x >= 0 and x < ww and y >= 0 and y < hh:
        color_vals[i, :] = jpg[y, x]


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

write_color_ply('color_pc.ply', points, color_vals)

# ccIntrinsics.height = 720
# ccIntrinsics.fy = 1042.0

# ux = 960
# uy = 540
# alpha = 1042.0
# gamma = 0
# K = np.array([[ux]])

# r_vec = ext_d[:3, :3]
# t_vec = -ext_d[:3, 3]

r_vec = ext_d[:3, :3]
t_vec = -ext_d[:3, 3]

k1, k2, k3 = get_k(calib)
im_coords, _ = cv2.projectPoints(points, r_vec, t_vec, intrinsic[:3, :3], np.array([k1, k2, 0, 0]))


color_vals = np.zeros_like(points)

for i, t in enumerate(im_coords):
    x, y = t.squeeze()
    x = int(np.round(x))
    y = int(np.round(y))
    if x >= 0 and x < ww and y >= 0 and y < hh:
        color_vals[i, :] = jpg[y, x]

write_color_ply('color_pc_cv2.ply', points, color_vals)