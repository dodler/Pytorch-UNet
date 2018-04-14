import torch.nn.functional as F
from utils import *

src_win = None
gt_win = None
pred_win = None

import random

def display_every_iter(iter_num, X, gt, prediction, vis):
    global src_win
    global gt_win
    global pred_win
    if iter_num % 50 == 0:
        img_index = random.randint(0,X.size()[0]-1)

        img = X.data.squeeze(0).cpu().numpy()[img_index]
        mask = gt.data.squeeze(0).cpu().numpy()[img_index]
        pred = (prediction > 0.6).float().data.squeeze(0).cpu().numpy()[img_index]
        pred[pred > 0] = 255
        mask[mask > 0] = 255

        if src_win is None:
            src_win = vis.image(img, opts=dict(title='source image'))
            gt_win = vis.image(mask, opts=dict(title='gt'))
            pred_win = vis.image(pred, opts=dict(title='prediction'))
        else:
            vis.image(img, opts=dict(title='source image'), win=src_win)
            vis.image(mask, opts=dict(title='gt'), win=gt_win)
            pred_win = vis.image(pred, opts=dict(title='prediction'), win=pred_win)
        # yy = dense_crf(np.array(prediction).astype(np.uint8), y)
