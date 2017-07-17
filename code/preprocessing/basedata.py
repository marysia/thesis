import os
import random
import numpy as np

class BaseData:
    def __init__(self, datadir = '/home/marysia/thesis/data'):
        self.datadir = datadir
        self._set_classes()
        self.load()

    def load(self):
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def shuffle(self, x, y):
        assert len(x) == len(y)
        p = np.random.permutation(len(x))
        return x[p], y[p]

    def _set_classes(self):
        raise NotImplementedError

    def _one_hot_encoding(self, array, nb_classes):
        '''
        Converts an array of class labels to a one-hot encoded matrix.
        E.g. [2, 1, 3] to [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        '''
        targets = array.astype(np.int)
        one_hot_targets = np.eye(nb_classes)[targets]
        return one_hot_targets


class Data:
    def __init__(self, scope, x, y):
        self.scope = scope
        self.x = x
        self.y = y

    def get_images(self):
        return self.x

    def get_labels(self):
        return self.y

    def get_sample(self):
        i = random.randint(0, self.x.shape[0])
        return Data(self.scope+'-sample', self.x[i], self.y[i])

    def get_batch(self, batch_size):
        size = self.x.shape[0]
        i = random.randint(0, self.x.shape[0])
        if i < (size - batch_size):
            return Data(self.scope+'-batch-'+str(batch_size), self.x[i:i+batch_size], self.y[i:i+batch_size])
        else:
            #
            end_set_x = self.x[i:]
            end_set_y = self.y[i:]
            begin_set_x = self.x[0:(batch_size-i)]
            begin_set_y = self.y[0:(batch_size-i)]

    def get_next_batch(self, i, batch_size):
        start = i*batch_size
        end = (i+1)*batch_size
        return Data(self.scope+str(batch_size)+'-batch-'+str(i), self.x[start:end], self.y[start:end])

    def shape(self):
        return self.x.shape, self.y.shape

