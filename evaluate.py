import sys
from optparse import OptionParser

from scipy.misc import imresize

from torch.optim.lr_scheduler import ReduceLROnPlateau

import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import *

from config import PER_ITER_IOU
from config import VAL_EPOCH_DICE
from config import PER_ITER_DICE
from config import VAL_EPOCH_BCE
from config import EPOCH_NUM
from config import PER_EPOCH_LOSS
from config import CHECKPOINT_DIR
from config import DATA
from config import PER_ITER_LOSS
from config import RESIZE_TO
from config import BATCH_SIZE
from config import gpu_id
from eval import eval_net
from unet import UNet
from utils import *
from utils.abstract import DualCompose
from utils.abstract import ImageOnly
from utils.dataset import InMemoryImgSegmDataset
from utils.dataset import JsonSegmentationDataset
from utils.dualcrop import DualRotatePadded
from utils.dualcolor import *
from utils.util_transform import DualResize, DualToTensor
from visualization.loss_watch import VisdomValueWatcher
from visualization.visdom_helper import display_every_iter

from myloss import dice_coeff

from config import RESTORE_FROM

watch = VisdomValueWatcher()

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

test_transform = DualCompose([
    DualResize((RESIZE_TO, RESIZE_TO)),
    DualToTensor(),
    ImageOnly(Normalize(rgb_mean, rgb_std))
])

jds = JsonSegmentationDataset(DATA, '/home/ubuntu/workdir/lyan/Pytorch-UNet/jsons/test.json', test_transform)
loader = DataLoader(jds, batch_size=1)

print('config batch size:',BATCH_SIZE)


def image_iou(predict_mask, mask, classes):
    
    thresh = 0.5
    predict_mask[predict_mask >thresh] = 1
    predict_mask[predict_mask <= thresh] = 0

    IOU = []
    for i in range(classes):
        intersection = ((mask == i) & (predict_mask == i)).sum()
        if intersection == 0:
            IOU.append(0)
            continue
        union = ((mask == i) | (predict_mask == i)).sum()
        if union == 0:
            IOU.append(-1)
            continue
        IOU.append(intersection / union)
    return np.mean(np.array(IOU))

def eval_net(net):
    net.eval()
    print('evaluation started')

    avg_iou = 0
    avg_dice = 0

    for i, b in tqdm(enumerate(loader)):

        X = b[0]
        y = b[1]
        if torch.cuda.is_available():
            X = Variable(X).cuda()
            y = Variable(y).cuda()
        else:
            X = Variable(X)
            y = Variable(y)

        probs = net(X)
        probs= probs.cpu().data.numpy().reshape((128,128))
 #       cv2.resize(probs, dsize=(40,40), interpolation=cv2.INTER_CUBIC)
        gt = y.cpu().data.numpy().reshape((128,128))
#        gt = cv2.resize(gt, dsize=(40,40), interpolation=cv2.NEAREST)

        avg_iou +=  image_iou(probs, gt, 2)

    print('avg_iou:',avg_iou / float(i))

if __name__ == '__main__':

    net = UNet(3, 1).cuda()
    net.load_state_dict(torch.load(RESTORE_FROM))
    print('Model loaded from {}'.format(RESTORE_FROM))

    eval_net(net)
