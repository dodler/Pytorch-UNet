import visdom


class VisdomValueWatcher(object):
    def __init__(self):
        self._watchers = {}
        self._vis = visdom.Visdom()

    def add_value(self, name, value):
        if name in self._watchers.keys():
            self._watchers[name].append(value)
        else:
            self._watchers[name] = [value]

    def output(self):
        pass

    def clean(self, name):
        self._watchers[name] = name
