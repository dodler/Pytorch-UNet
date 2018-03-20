import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn

import vis

from utils import dense_crf

from utils import *
from myloss import DiceLoss
from eval import eval_net
from unet import UNet
from torch.autograd import Variable
from torch import optim
from optparse import OptionParser
import sys
import os
import os.path as osp
from torch.nn import Upsample

from config import gpu_id
from config import DATA

print(gpu_id)
print(DATA)

def train_net(net, epochs=5, batch_size=8, lr=0.1, val_percent=0.05,
              cp=True, gpu=False):
    dir_img = osp.join(DATA,'images/')
    dir_mask = osp.join(DATA, 'mask/')
    dir_checkpoint = 'checkpoints/'

    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(cp), str(gpu)))

    N_train = len(iddataset['train'])

    train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask)
    val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask)

    optimizer = optim.SGD(net.parameters(),
                          lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.BCELoss()

    epoch_dices = []
    epoch_losses = []
    epochs_p = []

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch+1, epochs))
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask)

        epoch_loss = 0
        it_losses = []
        its = []
#        if 1:
#            val_dice = eval_net(net, val, gpu)
#            print('Validation Dice Coeff: {}'.format(val_dice))

        for i, b in enumerate(batch(train, batch_size)):
            X = np.array([i[0] for i in b])
            y = np.array([i[1] for i in b])

            X = torch.FloatTensor(X)
            y = torch.ByteTensor(y)

            if gpu:
                X = Variable(X).cuda(gpu_id)
                y = Variable(y).cuda(gpu_id)
            else:
                X = Variable(X)
                y = Variable(y)

            y_pred = net(X)

            probs = F.sigmoid(y_pred)
            probs_flat = probs.view(-1)

            if i % 50 == 0:
                img = X.data.squeeze(0).cpu().numpy()[0]
#                img = np.transpose(img, axes=[1, 2, 0])
                mask = y.data.squeeze(0).cpu().numpy()[0]
                pred = (F.sigmoid(y_pred) > 0.8).float().data.squeeze(0).cpu().numpy()[0]
#                Q = dense_crf(((img*255).round()).astype(np.uint8), pred)
                vis.show(img, mask, pred, 'image - mask - predict - densecrf')

            y_flat = y.view(-1)

            loss = criterion(probs_flat, y_flat.float())
            epoch_loss += loss.data[0]

            it_losses.append(loss.data[0])

            its.append(i)

            vis.plot_loss(np.array(its), np.array(it_losses), 'iteration losses')

            print('{0:.4f} --- loss: {1:.6f}'.format(i*batch_size/N_train,
                                                     loss.data[0]))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss/i))

        epochs_p.append(e)
        epoch_losses.append(np.mean(np.array(it_losses)))
        vis.plot_loss(np.array(epochs_p), np.array(epoch_losses), 'epoch losses')

        val_dice = eval_net(net, val, gpu)

        vis.plot_loss(epoch_dices, 'epoch dices')
        print('Validation Dice Coeff: {}'.format(val_dice))


        if cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}_loss{}.pth'.format(epoch+1, loss.data[0]))

            print('Checkpoint {} saved !'.format(epoch+1))


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

    if options.gpu:
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
