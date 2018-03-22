import os
import os.path as osp

from PIL import Image

from config import DATA

files = os.listdir(osp.join(DATA, 'mask'))

files = [k for k in files if '.ppm' in k]

for f in files:
    img = Image.open(osp.join(DATA, 'mask', f))
    img.save(osp.join(DATA, 'mask', f.split('.')[0] + '.png'), 'PNG')
