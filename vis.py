import visdom
import numpy as np

vis = visdom.Visdom()
vis.text('Visualizing segmentation output')

def show(pred, labels):
    print(pred.shape, labels.shape)
    vis.images(np.hstack([pred, labels]))

def show(pred, labels, title):
    print(pred.shape, labels.shape)
    vis.images(np.hstack([pred,labels]),opts=dict(title=title, caption='How random.'))

