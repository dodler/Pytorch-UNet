import visdom
import numpy as np


class VisdomValueWatcher(object):
    def __init__(self):
        self._watchers = {}
        self._wins = {}
        self._vis = visdom.Visdom()

    def get_vis(self):
        return self._vis

    def add_value(self, name, value):
        if name in self._watchers.keys():
            self._watchers[name].append(value)
        else:
            self._watchers[name] = [value]

    def output(self):
        for name in self._wins.keys():
            self._vis.line(Y=np.array(self._watchers[name]),
                           X=np.array(range(len(self._watchers[name]))),
                           win=self._wins[name], update='append',
                           opts=dict(title=name))

    def output(self, name):
        if name in self._wins.keys():
            self._vis.line(Y=np.array(self._watchers[name]),
                           X=np.array(range(len(self._watchers[name]))),
                           win=self._wins[name], update='append',
                           opts=dict(title=name))
        else:
            self._wins[name] = self._vis.line(Y=np.array(self._watchers[name]),
                                              X=np.array(range(len(self._watchers[name]))),
                                              opts=dict(title=name))

    def clean(self, name):
        self._watchers[name] = name
