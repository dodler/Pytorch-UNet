import torch.nn.functional as F
from utils import *


def display_every_iter(iter_num, X, gt, prediction, vis):
    if iter_num % 50 == 0:
        img = X.data.squeeze(0).cpu().numpy()[0]
        #                img = np.transpose(img, axes=[1, 2, 0])
        mask = gt.data.squeeze(0).cpu().numpy()[0]
        pred = (prediction > 0.6).float().data.squeeze(0).cpu().numpy()[0]
        #                Q = dense_crf(((img*255).round()).astype(np.uint8), pred)

        vis.image(img, opts=dict(title='source image'))
        vis.image(mask, opts=dict(title='gt'))
        vis.image(pred, opts=dict(title='prediction'))
        # yy = dense_crf(np.array(prediction).astype(np.uint8), y)
