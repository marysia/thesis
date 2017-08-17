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
    def __init__(self, scope, x, y, id, balanced):
        self.scope = scope
        self.x = x
        self.y = y
        self.id = id
        self.balanced = balanced
        self.samples = len(x)

        if not self.balanced:
            self.set_balanced_values()

    def get_images(self):
        return self.x

    def get_labels(self):
        return self.y

    def get_sample(self):
        i = random.randint(0, self.x.shape[0])
        return Data(self.scope+'-sample', self.x[i], self.y[i])

    def get_next_batch(self, i, batch_size):
        if self.balanced:
            return self.get_balanced_batch(i, batch_size)
        else:
            return self.get_unbalanced_batch(i, batch_size)

    def get_balanced_batch(self, i, batch_size):
        start = i*batch_size
        end = (i+1)*batch_size

        return self.x[start:end], self.y[start:end]

    def get_unbalanced_batch(self, i, batch_size):
        f = int(batch_size / self.fraction)

        # positive part
        start = i * f
        end = (i + 1) * f
        pos_idx = self.pos_idx[start:end]

        # negative part
        random_idx = np.random.randint(0, len(self.neg_idx), (batch_size - f,))
        neg_idx = self.neg_idx[random_idx]

        # create x, y
        x = np.concatenate([self.x[pos_idx], self.x[neg_idx]])
        y = np.concatenate([self.y[pos_idx], self.y[neg_idx]])
        p = np.random.permutation(len(x))
        return x[p], y[p]

    def set_balanced_values(self):
        self.pos_idx = [i for i, elem in enumerate(self.y) if np.argmax(elem) == 1]
        self.neg_idx = [i for i, elem in enumerate(self.y) if np.argmax(elem) == 0]
        self.fraction = 5


    def shape(self):
        return self.x.shape, self.y.shape
