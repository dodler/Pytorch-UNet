import visdom
import numpy as np

vis = visdom.Visdom()
vis.text('Visualizing segmentation output')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

lines = {}

def plot_loss(it, loss, label):
    if label in lines:
        vis.line(X=it, Y=loss, name=label, win=lines[label],update='new')
    else:
        lines[label] = vis.line(X=it, Y=loss, name=label)

def show(image, label):
    vis.image(image, title=label)

def show(pred, labels):
    print(pred.shape, labels.shape)
    vis.images(np.hstack([pred, labels]))

def show(pred, labels, title):
    print(pred.shape, labels.shape)
    vis.images(np.hstack([pred,labels]),opts=dict(title=title, caption='How random.'))

def show_predict(pred, title):
    vis.images(pred, title)

def show(img, mask, pred, title):
    print(img.shape, mask.shape, pred.shape)
    vis.image(img , opts=dict(title='source image'))
    vis.image(mask, opts=dict(title='gt'))
    vis.image(pred, opts=dict(title='prediction'))

