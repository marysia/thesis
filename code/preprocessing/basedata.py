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
        return x[p], y[p], p

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

    def get_next_batch(self, i, batch_size):
        start = i*batch_size
        end = (i+1)*batch_size
        return Data(self.scope+str(batch_size)+'-batch-'+str(i), self.x[start:end], self.y[start:end])

    def shape(self):
        return self.x.shape, self.y.shape

class UnbalancedData:
    def __init__(self, scope, x_pos, y_pos, x_neg, y_neg):
        self.scope = scope
        self.x = x_pos
        self.x_neg = x_neg

        self.y_pos = y_pos
        self.y_neg = y_neg

        self.fraction = 10

    def get_next_batch(self, i, batch_size):
        fraction = batch_size / 2

        start = i*fraction
        end = (i+1)*fraction

        x_pos = self.x[start:end]
        x_neg = self.x_neg[start:end]


        x = np.concatenate([x_pos, x_neg])
        y = np.concatenate([self.y_pos[0:fraction], self.y_neg[0:fraction]])
        p = np.random.permutation(len(x))
        return Data('%s-%d-batch-%d' % (self.scope, batch_size, i), x[p], y[p])

    def get_next_batch2(self, i, batch_size):
        fraction = int(batch_size / self.fraction)

        # get positive part
        start = i*fraction
        end = (i+1)*fraction
        x_pos = self.x[start:end]

        # get negative part
        rand_idx = np.random.randint(0, x_pos.shape[0], (batch_size - fraction,))
        x_neg = self.x_neg[rand_idx]

        x = np.concatenate([x_pos, x_neg])
        y = np.concatenate([self.y_pos[0:fraction], self.y_neg[0:batch_size - fraction]])
        p = np.random.permutation(len(x))
        return Data('%s-%d-batch-%d' % (self.scope, batch_size, i), x[p], y[p])



