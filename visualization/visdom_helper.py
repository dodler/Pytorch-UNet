import torch.nn.functional as F


def display_every_iter(iter_num, X, gt, prediction, vis):
    if iter_num % 50 == 0:
        img = X.data.squeeze(0).cpu().numpy()[0]
        #                img = np.transpose(img, axes=[1, 2, 0])
        mask = gt.data.squeeze(0).cpu().numpy()[0]
        pred = (F.sigmoid(prediction) > 0.8).float().data.squeeze(0).cpu().numpy()[0]
        #                Q = dense_crf(((img*255).round()).astype(np.uint8), pred)
        vis.show(img, mask, pred, 'image - mask - predict - densecrf')
