import sys
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Resize
from tqdm import *

import vis
from config import DATA
from config import gpu_id
from config import CHECKPOINT_DIR
from config import RESIZE_TO
from eval import eval_net
from unet import UNet
from utils import *

from utils.dualcrop import DualRotatePadded
from utils.abstract import DualCompose, Dualized
from utils.dataset import InMemoryImgSegmDataset
from utils.abstract import ImageOnly
from utils.util_transform import DualResize, DualToTensor

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
loader = DataLoader(dataset, batch_size=1)

train_len = len(dataset)
dataset.set_mode('val')
val_len = len(dataset)
dataset.set_mode('train')


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

    epoch_dices = []
    epoch_losses = []
    epochs_p = []

    for epoch_num in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch_num + 1, epochs))

        epoch_loss = 0
        it_losses = []
        its = []
        dataset.set_mode('train')
        for i, b in tqdm(enumerate(loader)):
            print('batch',len(b))

            X = b[0]
            y = b[1]

            if gpu and torch.cuda.is_available():
                X = Variable(X).cuda(gpu_id)
                y = Variable(y).cuda(gpu_id)
            else:
                X = Variable(X)
                y = Variable(y)

            print('doing forward')
            y_pred = net(X)

            probs = F.sigmoid(y_pred)
            probs_flat = probs.view(-1)

            # if i % 50 == 0:
            #     img = X.data.squeeze(0).cpu().numpy()[0]
            #     #                img = np.transpose(img, axes=[1, 2, 0])
            #     mask = y.data.squeeze(0).cpu().numpy()[0]
            #     pred = (F.sigmoid(y_pred) > 0.8).float().data.squeeze(0).cpu().numpy()[0]
            #     #                Q = dense_crf(((img*255).round()).astype(np.uint8), pred)
            #     vis.show(img, mask, pred, 'image - mask - predict - densecrf')

            y_flat = y.view(-1)

            loss = criterion(probs_flat, y_flat.float())
            # epoch_loss += loss.data[0]

            # it_losses.append(loss.data[0])

            # its.append(i)

            # vis.plot_loss(np.array(its), np.array(it_losses), 'iteration losses')

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train,
                                                     loss.data[0]))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        epochs_p.append(epoch_num)
        epoch_losses.append(np.mean(np.array(it_losses)))
        vis.plot_loss(np.array(epochs_p), np.array(epoch_losses), 'epoch losses')

        dataset.set_mode('val')
        val_dice = eval_net(net, dataset, gpu)

        # vis.plot_loss(epoch_dices, 'epoch dices')
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

    (options, args) = parser.parse_args()

    net = UNet(3, 1)

    #    if options.load:
    #    net.load_state_dict(torch.load('INTERRUPTED.pth'))
    #    print('Model loaded from {}'.format('interrupted.pth'))

    if options.gpu and torch.cuda.is_available():
        net.cuda(gpu_id)
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
