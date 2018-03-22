import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import *

from config import USE_MATPLOTLIB_VIS
from myloss import dice_coeff
from utils import dense_crf

from config import gpu_id

def eval_net(net, dataset, gpu=False):
    tot = 0
    for i, b in tqdm(enumerate(dataset)):
        X = b[0]
        y = b[1]

        X = torch.FloatTensor(X).unsqueeze(0)
        y = torch.ByteTensor(y).unsqueeze(0)

        if gpu:
            X = Variable(X, volatile=True).cuda(gpu_id[0])
            y = Variable(y, volatile=True).cuda(gpu_id[0])
        else:
            X = Variable(X, volatile=True)
            y = Variable(y, volatile=True)

        y_pred = net(X)

        y_pred = (F.sigmoid(y_pred) > 0.3).float()
        #        print(y_pred.size())
        #        print(y.size())

        dice = dice_coeff(y_pred, y.float()).data[0]
        tot += dice

        if USE_MATPLOTLIB_VIS:
            X = X.data.squeeze(0).cpu().numpy()
            X = np.transpose(X, axes=[1, 2, 0])
            y = y.data.squeeze(0).cpu().numpy()
            y_pred = y_pred.data.squeeze(0).squeeze(0).cpu().numpy()

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 4, 1)
            ax1.imshow(X)
            ax2 = fig.add_subplot(1, 4, 2)
            ax2.imshow(y)
            ax3 = fig.add_subplot(1, 4, 3)
            ax3.imshow((y_pred > 0.5))

            Q = dense_crf(((X * 255).round()).astype(np.uint8), y_pred)
            ax4 = fig.add_subplot(1, 4, 4)
            print(Q)
            ax4.imshow(Q > 0.5)
            plt.show()
    return tot / i
