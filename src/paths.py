import os

png_128_path = r'/home/sandro/Dokumente/WHo/png128/'

PNG_FILES = None
# PNG_FILES = os.listdir(png_128_path)
# PNG_FILES = list(filter(lambda x: '.png' in x, PNG_FILES))
# PNG_FILES = list(map(lambda x: os.path.join(png_128_path, x), PNG_FILES))


PCD_PATH = r'/home/sandro/Dokumente/WHo/PCD/'
import glob
PCD_FILES = sorted(glob.glob(os.path.join(PCD_PATH, '*', '*', 'pcd', '*.pcd')))


if __name__ == '__main__':
    # print(PNG_FILES)
    print(PCD_FILES)