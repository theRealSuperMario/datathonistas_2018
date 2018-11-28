import cv2
import numpy as np

from pylab import *
import sys
sys.path.insert(0, '..')
import pandas as pd
from src.paths import PNG_FILES, PCD_FILES
from src.cgmcore.utils import load_pcd_as_ndarray, pointcloud_to_rgb_map2, show_rgb_map
import cv2
import os
from src.segmentation import SegmentationData, transfer_segmentation, transfer_image, segment_foreground, maybe_roll_seg_image

from pylab import *
from pyntcloud import PyntCloud
from pyntcloud.io import write_ply


## MH_WHH_0003
# pcd_file = r'/home/sandro/Dokumente/WHo/PCD/MH_WHH_0003/1536914114096/pcd/pc_MH_WHH_0003_1536913710517_104_000.pcd'
# jpg_file = r'/home/sandro/Dokumente/WHo/PCD/MH_WHH_0003/1536914114096/jpg/rgb_MH_WHH_0003_1536913710517_104_7544.033428759.jpg'
# seg_file = r'/home/sandro/Dokumente/WHo/denseposeoutput_first10/rgb_MH_WHH_0003_1536913710517_104_7544.033428759_INDS.png'

## MH_WHH_0004
# pcd_file = r'/home/sandro/Dokumente/WHo/PCD/MH_WHH_0004/1536914114092/pcd/pc_MH_WHH_0004_1536913535697_104_000.pcd'
# jpg_file = r'/home/sandro/Dokumente/WHo/PCD/MH_WHH_0004/1536914114092/jpg/rgb_MH_WHH_0004_1536913535697_104_7355.294470759.jpg'
# seg_file = r'/home/sandro/Dokumente/WHo/denseposeoutput_first10/rgb_MH_WHH_0004_1536913535697_104_7355.294470759_INDS.png'


## MH_WHH_0008
# pcd_file = r'/home/sandro/Dokumente/WHo/PCD/MH_WHH_0008/1537167614128/pcd/pc_MH_WHH_0008_1537167343405_104_000.pcd'
# jpg_file = r'/home/sandro/Dokumente/WHo/PCD/MH_WHH_0008/1537167614128/jpg/rgb_MH_WHH_0008_1537167343405_104_2077.545678759.jpg'
# seg_file = r'/home/sandro/Dokumente/WHo/denseposeoutput_first10/rgb_MH_WHH_0008_1537167343405_104_2077.545678759_INDS.png'

## MH_WHH_0010
pcd_file = r'/home/sandro/Dokumente/WHo/PCD/MH_WHH_0010/1537167314100/pcd/pc_MH_WHH_0010_1537166990387_104_000.pcd'
jpg_file = r'/home/sandro/Dokumente/WHo/PCD/MH_WHH_0010/1537167314100/jpg/rgb_MH_WHH_0010_1537166990387_104_1719.812364759.jpg'
seg_file = r'/home/sandro/Dokumente/WHo/denseposeoutput_first10/rgb_MH_WHH_0010_1537166990387_104_1719.812364759_INDS.png'


cloud = PyntCloud.from_file(pcd_file)  # :PyntCloud
rgb_image = cv2.imread(jpg_file, -1).astype(np.uint8)[::-1, :, :]
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

seg_image = cv2.imread(seg_file, -1).astype(np.uint8)

seg_image = maybe_roll_seg_image(rgb_image, seg_image)
seg_image = seg_image[::-1, :]



points = cloud.points.values[:, :3]

seg_vals = transfer_segmentation(seg_image, points)
color_vals = transfer_image(rgb_image, points)


from src.segmentation import write_color_ply, colorize_seg_vals


base_name = os.path.basename(pcd_file)

color_seg_vals = colorize_seg_vals(seg_vals)
write_color_ply('{}_color_seg.ply'.format(base_name), points, color_seg_vals)
write_color_ply('{}_rgb.ply'.format(base_name), points, color_vals)
write_color_ply('{}_just_points.ply'.format(base_name), points, np.ones_like(points)*255)

fg_array = segment_foreground(pcd_file, jpg_file, seg_file, save_ply=False)

write_color_ply('{}_fg.ply'.format(base_name), fg_array, np.ones_like(fg_array))






