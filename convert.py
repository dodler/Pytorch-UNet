import os
import os.path as osp
import PIL
from PIL import Image

DATA = '/home/ubuntu/recognition/data/masks_clean/HAIR/mask/'

files = os.listdir(DATA)

files = [k for k in files if '.jpg' in k]

for f in files:
    img = Image.open(osp.join(DATA, f))
    img.save(osp.join(DATA, f.split('.')[0] + '.png'), 'PNG')
