import sys

from src.segmentation import write_color_ply
from src.tango import get_intrinsic_matrix, get_k, get_extrinsic_matrix

sys.path.insert(0, '..')
import cv2

from pylab import *
from pyntcloud import PyntCloud

pcd_file = r'/home/sandro/Dokumente/WHo/PCD/1536913514153/pcd/pc_MH_WHH_0001_1536913319075_104_000.pcd'
# jpg_file = r'/home/sandro/Dokumente/WHo/PCD/1536911714062/jpg/rgb_MH_WHH_0014_1536911466814_104_5328.379994052.jpg'
jpg_file = r'/home/sandro/Dokumente/WHo/PCD/1536913514153/jpg/rgb_MH_WHH_0001_1536913319075_104_7137.838129759.jpg'

calib_file = r'calibration.xml'


cloud = PyntCloud.from_file(pcd_file)  # :PyntCloud

jpg = cv2.imread(jpg_file, -1)
jpg = jpg

hh, ww, _ = jpg.shape

points = cloud.points.values[:, :3]

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