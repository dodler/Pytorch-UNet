import sys
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import *

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
from utils.util_transform import DualResize, DualToTensor
from visualization.loss_watch import VisdomValueWatcher
from visualization.visdom_helper import display_every_iter

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

test_transform = DualCompose([
    DualResize((RESIZE_TO, RESIZE_TO)),
    DualToTensor(),
    ImageOnly(Normalize(rgb_mean, rgb_std))
])

dataset = InMemoryImgSegmDataset(DATA,
                                 'original', 'mask',
                                 train_transform, test_transform,
                                 limit_len=50)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

train_len = len(dataset)
dataset.set_mode('val')
val_len = len(dataset)
dataset.set_mode('train')

def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)

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
    criterion = nn.BCELoss()

    for epoch_num in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch_num + 1, epochs))

        epoch_loss = 0
        dataset.set_mode('train')
        for i, b in tqdm(enumerate(loader)):

            X = b[0]
            y = b[1]

            if gpu and torch.cuda.is_available():
                X = Variable(X).cuda(gpu_id[0])
                y = Variable(y).cuda(gpu_id[0])
            else:
                X = Variable(X)
                y = Variable(y)

            probs = F.sigmoid(net(X))
            probs_flat = probs.view(-1)

            y_flat = y.view(-1)

            display_every_iter(X, i, y, probs, watch.get_vis())
            loss = criterion(probs_flat, y_flat.float())

            watch.add_value(PER_ITER_LOSS, loss.cpu().data[0])

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / len(dataset),
                                                     loss.data[0]))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        watch.output()
        dataset.set_mode('val')
        net.eval()
        val_dice = eval_net(net, dataset, gpu)
        net.train()

        print('Validation Dice Coeff: {}'.format(val_dice))

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

    net = nn.DataParallel(UNet(3, 1), device_ids=gpu_id)

    if options.restore:
        net.load_state_dict(torch.load('INTERRUPTED.pth'))
        print('Model loaded from {}'.format('interrupted.pth'))

    if options.gpu and torch.cuda.is_available():
        net.cuda()
        cudnn.benchmark = True

    try:
        train_net(net, options.epochs, options.batchsize, options.lr,
                  gpu=options.gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
