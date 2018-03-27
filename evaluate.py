import sys
from optparse import OptionParser

from torch.optim.lr_scheduler import ReduceLROnPlateau

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

print(gpu_id)
print(DATA)

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

test_transform = DualCompose([
    DualResize((RESIZE_TO, RESIZE_TO)),
    DualToTensor(),
    ImageOnly(Normalize(rgb_mean, rgb_std))
])

jds = JsonSegmentationDataset(DATA, '/home/ubuntu/workdir/lyan/Pytorch-UNet/jsons/test.json', test_transform)
loader = DataLoader(jds, batch_size=BATCH_SIZE)

print('config batch size:',BATCH_SIZE)

def batch_iou(pred,target, n_classes=2):
    blen = pred.size()[0]
    riou = 0
    for i in range(blen):
        riou += iou(pred[i,:],target[i,:], n_classes)
    return riou / blen

def iou(pred, target, n_classes = 1):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    pred_inds = pred > 0.6
    target_inds = target == 1
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    return float(intersection) / (1e-05 + float(max(union, 1)))

upsample = torch.nn.Upsample(size=(RESIZE_TO, RESIZE_TO))
sigmoid = torch.nn.Sigmoid()

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


def eval_net(net, epochs=5, batch_size=8):
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
        probs_flat = probs.view(-1)
        y_flat = y.view(-1)
        display_every_iter(i, X, y, probs, watch.get_vis())

        dice = dice_coeff(probs_flat, y.float()).data[0]
        iou_m = batch_iou(probs, y, 2)

        avg_iou += iou_m
        avg_dice += dice

        watch.add_value(PER_ITER_IOU, iou_m)
        watch.output(PER_ITER_IOU)
        watch.add_value(PER_ITER_DICE, dice)
        watch.output(PER_ITER_DICE)
    print('avg_iou:',avg_iou / float(i), ' avg_dice',avg_dice / float(i))

if __name__ == '__main__':

    net = UNet(3, 1).cuda()
    net.load_state_dict(torch.load(RESTORE_FROM))
    print('Model loaded from {}'.format(RESTORE_FROM))

    try:
        eval_net(net, EPOCH_NUM, BATCH_SIZE)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
