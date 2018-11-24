import cv2
import numpy as np

from pylab import *
import sys
sys.path.insert(0, '..')
import pandas as pd
from src.paths import PNG_FILES, PCD_FILES
from src.cgmcore.utils import load_pcd_as_ndarray, pointcloud_to_rgb_map2, show_rgb_map
import cv2

from src.segmentation import SegmentationData, transfer_segmentation, transfer_image

from pylab import *
from pyntcloud import PyntCloud
from pyntcloud.io import write_ply

pcd_file = r'/home/sandro/Dokumente/WHo/PCD/MH_WHH_0008/1537167614128/pcd/pc_MH_WHH_0008_1537167343405_104_001.pcd'
# jpg_file = r'/home/sandro/Dokumente/WHo/PCD/1536911714062/jpg/rgb_MH_WHH_0014_1536911466814_104_5328.379994052.jpg'
mh_num = 8
time_stamp = 1537167343405
some_number = 104
idx = 2078
jpg_file = r'/home/sandro/Dokumente/WHo/PCD/MH_WHH_0008/1537167614128/jpg/rgb_MH_WHH_0008_1537167343405_104_2078.081837052.jpg'
seg_data = SegmentationData()
seg_file = seg_data.get_inds(mh_num, time_stamp, some_number, idx)

cloud = PyntCloud.from_file(pcd_file)  # :PyntCloud
rgb_image = cv2.imread(jpg_file, -1).astype(np.uint8)[::-1, :, :]
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
seg_image = cv2.imread(seg_file, -1).astype(np.uint8)
seg_image = np.rollaxis(seg_image, 1, 0)
points = cloud.points.values[:, :3]

seg_vals = transfer_segmentation(seg_image, points)
color_vals = transfer_image(rgb_image, points)


from src.segmentation import write_color_ply, colorize_seg_vals

color_seg_vals = colorize_seg_vals(seg_vals)
write_color_ply('wh_008_104_2078_color_seg.ply', points, color_seg_vals)
write_color_ply('wh_008_104_2078_rgb.ply', points, color_vals)


fg_array = points[seg_vals == 1]
np.savetxt('pc_MH_WHH_0008_1537167343405_104_001.txt', fg_array)







