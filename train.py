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
from utils.dualcrop import DualRotatePadded
from utils.dualcolor import *
from utils.util_transform import DualResize, DualToTensor
from visualization.loss_watch import VisdomValueWatcher
from visualization.visdom_helper import display_every_iter

from myloss import dice_coeff

watch = VisdomValueWatcher()

print(gpu_id)
print(DATA)

rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)

train_transform = DualCompose([
    DualResize((RESIZE_TO, RESIZE_TO)),
    DualRotatePadded(30),
    DualToTensor(),
    ImageOnly(Normalize(rgb_mean, rgb_std))])
#    ImageOnly(RandomSaturation(-0.1,0.1)),
#    ImageOnly(RandomGamma(0.9,1.1))])

test_transform = DualCompose([
    DualResize((RESIZE_TO, RESIZE_TO)),
    DualToTensor(),
    ImageOnly(Normalize(rgb_mean, rgb_std))
])

dataset = InMemoryImgSegmDataset(DATA,
                                 'original','mask',
                                 train_transform, test_transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

train_len = len(dataset)
dataset.set_mode('val')
val_len = len(dataset)
dataset.set_mode('train')

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

def train_net(net, epochs=5, batch_size=8, lr=0.1, cp=True, gpu=False):
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, train_len, val_len, str(cp), str(gpu)))

    optimizer = optim.SGD(net.parameters(),
                          lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.BCELoss().cuda()
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    for epoch_num in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch_num + 1, epochs))

        epoch_loss = 0
        dataset.set_mode('train')
        for i, b in tqdm(enumerate(loader)):

            X = b[0]
            y = b[1]

            if gpu and torch.cuda.is_available():
                X = Variable(X).cuda()
                y = Variable(y).cuda()
            else:
                X = Variable(X)
                y = Variable(y)

            probs = net(X)
            probs_flat = probs.view(-1)
            y_flat = y.view(-1)

            display_every_iter(i, X, y, probs, watch.get_vis())

            loss = criterion(probs_flat, y_flat.float())
            epoch_loss += loss.cpu().data[0]

            dice = dice_coeff(probs_flat, y.float()).data[0]
            iou_m = batch_iou(probs, y, 2)

            watch.add_value(PER_ITER_IOU, iou_m)
            watch.output(PER_ITER_IOU)
            watch.add_value(PER_ITER_LOSS, loss.cpu().data[0])
            watch.output(PER_ITER_LOSS)
            watch.add_value(PER_ITER_DICE, dice)
            watch.output(PER_ITER_DICE)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

#        watch.clean(PER_ITER_LOSS)
#        watch.clean(PER_ITER_IOU)
#        watch.clean(PER_ITER_DICE)

        watch.add_value(PER_EPOCH_LOSS, epoch_loss / float(i))
        watch.output(PER_EPOCH_LOSS)
        print('Epoch finished ! Loss: {}'.format(epoch_loss / float(i)))

        dataset.set_mode('val')
        net.eval()
        val_dice, val_bce = eval_net(net, dataset, gpu)
        watch.add_value(VAL_EPOCH_DICE, val_dice)
        watch.add_value(VAL_EPOCH_BCE, val_bce)
        watch.output(VAL_EPOCH_DICE)
        watch.output(VAL_EPOCH_BCE)
#        print(val_dice, val_bce)
        scheduler.step(val_bce)
        net.train()

        print('Validation Dice Coeff: {}, bce: {}'.format(val_dice, val_bce))

        if cp:
            torch.save(net.state_dict(),
                       CHECKPOINT_DIR + 'CP{}_loss{}.pth'.format(epoch_num + 1, loss.data[0]))

            print('Checkpoint {} saved !'.format(epoch_num + 1))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=30, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=1e-3,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-r', '--restore', dest='restore', default=False,
                      help='Restore model from saved INTERRUPTED.pth')

    (options, args) = parser.parse_args()

#    net = nn.DataParallel(UNet(3, 1).cuda())
    net = UNet(3, 1).cuda()

#    if options.restore:
    net.load_state_dict(torch.load('INTERRUPTED.pth'))
    print('Model loaded from {}'.format('interrupted.pth'))

#    if options.gpu and torch.cuda.is_available():
#        net.cuda()
#        cudnn.benchmark = True

    try:
        train_net(net, EPOCH_NUM, options.batchsize, options.lr,
                  gpu=options.gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
