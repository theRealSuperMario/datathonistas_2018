import cv2
import numpy as np

from pylab import *
import sys
sys.path.insert(0, '..')

from src.paths import PNG_FILES, PCD_FILES
from src.cgmcore.utils import load_pcd_as_ndarray, pointcloud_to_rgb_map2, show_rgb_map
import cv2

from pylab import *
from pyntcloud import PyntCloud
from pyntcloud.io import write_ply

pcd_file = r'/home/sandro/Dokumente/WHo/PCD/1536911714062/pcd/pc_MH_WHH_0014_1536911466814_104_000.pcd'
jpg_file = r'/home/sandro/Dokumente/WHo/PCD/1536911714062/jpg/rgb_MH_WHH_0014_1536911466814_104_5328_379994052.jpg'

cloud = PyntCloud.from_file(pcd_file)  # :PyntCloud
write_ply('pc_MH_WHH_0008_1537167343405_104_000.ply', points=cloud.points, as_text=True)


jpg = cv2.imread(jpg_file, -1)
plt.imshow(jpg)


