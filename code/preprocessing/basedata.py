import random

import numpy as np


class BaseData:
    def __init__(self, datadir='/home/marysia/thesis/data'):
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
    def __init__(self, scope, x, y, nb_classes, balanced):
        self.scope = scope
        self.data = x
        self.labels = self.one_hot_encoding(y, nb_classes)
        self.nb_classes = nb_classes
        self.balanced = balanced
        self.samples = len(x)

        self.x, self.y , self.id = self.shuffle(self.data, self.labels)


    def shuffle(self, x, y):
        '''
        Shuffle dataset (x and y) by getting a random permutation of the indices.
        '''
        assert len(x) == len(y)
        p = np.random.permutation(len(x))
        return x[p], y[p], p

    def one_hot_encoding(self, array, nb_classes):
        '''
        One-hot encoding of the labels. E.g. [2, 0, 1] becomes [[0 0 1], [1 0 0], [0 1 0]]
        '''
        targets = array.astype(np.int)
        one_hot_targets = np.eye(nb_classes)[targets]
        return one_hot_targets

    def resize(self, samples):
        '''
        Resize the dataset to smaller dataset of _samples_ samples, but ensure it always happens in the same way
        by picking the first _samples_ elements from the original x and y data and returning those, shuffled.
        '''
        samples_per_class = samples / self.nb_classes

        x = []
        y = []
        for i in xrange(self.nb_classes):
            target_encoding = self.one_hot_encoding(np.array([i]), self.nb_classes)
            index = np.where(self.labels == target_encoding)[0][0]
            x.append(self.data[index:(index + samples_per_class)])
            y.append(self.labels[index:(index + samples_per_class)])

        x = np.concatenate(x)
        y = np.concatenate(y)

        if samples % self.nb_classes == 0:
            assert len(x) == samples
            assert len(y) == samples
            assert len(set(sum(y))) == 1
        else:
            assert len(x) == (samples - (samples % self.nb_classes))
            assert len(y) == (samples - (samples % self.nb_classes))
            assert len(set(sum(y))) == 1
            assert (sum(y)[0] * len(sum(y))) + (samples % self.nb_classes) == samples

        self.x, self.y, self.id = self.shuffle(x, y)
        self.samples = len(self.x)

    def get_next_batch(self, i, batch_size):
        ''' Get the next batch, based on whether the data is balanced or unbalanced.'''
        if self.balanced:
            return self.get_balanced_batch(i, batch_size)
        else:
            return self.get_unbalanced_batch(i, batch_size)

    def get_balanced_batch(self, i, batch_size):
        ''' Get the next balanced batch. In case the next batch cannot be retrieved, get random indices.'''
        start = i * batch_size
        end = (i + 1) * batch_size
        if len(self.x) > end:
            return self.x[start:end], self.y[start:end]
        else:
            p = np.random.permutation(len(self.x))
            indices = p[: (batch_size - (len(self.x) - start))]

            x = np.concatenate([self.x[start:], self.x[indices]])
            y = np.concatenate([self.y[start:], self.y[indices]])

            return x, y



class Data_old:
    def __init__(self, scope, x, y, id, balanced):
        self.scope = scope
        self.x = x
        self.y = y
        self.id = id
        self.balanced = balanced
        self.samples = len(x)

        if not self.balanced:
            self.set_balanced_values()

    def resize(self, samples):

        if not samples > len(self.x):
            if self.balanced:
                self.x = self.x[:samples]
                self.y = self.y[:samples]
                self.id = self.id[:samples]
                self.samples = len(self.x)

            else:
                pos_number = int(samples / (len(self.neg_idx) / float(len(self.pos_idx))))
                self.pos_idx = self.pos_idx[:pos_number]
                self.neg_idx = self.neg_idx[:(samples-pos_number)]

                self.pos_step = 0
                self.neg_step = 0
                self.samples = len(self.pos_idx) + len(self.neg_idx)

    def get_images(self):
        return self.x

    def get_labels(self):
        return self.y

    def get_sample(self):
        i = random.randint(0, self.x.shape[0])
        return Data(self.scope + '-sample', self.x[i], self.y[i])

    def get_next_batch(self, i, batch_size):
        if self.balanced:
            return self.get_balanced_batch(i, batch_size)
        else:
            return self.get_unbalanced_batch(i, batch_size)

    def get_balanced_batch(self, i, batch_size):
        start = i * batch_size
        end = (i + 1) * batch_size
        if len(self.x) > end:
            return self.x[start:end], self.y[start:end]
        else:
            p = np.random.permutation(len(self.x))
            indices = p[: (batch_size - (len(self.x) - start))]

            x = np.concatenate([self.x[start:], self.x[indices]])
            y = np.concatenate([self.y[start:], self.y[indices]])

            return x, y

    def get_unbalanced_batch(self, i, batch_size):

        idx_pos = self.pos_idx[self.pos_step*(batch_size / 2):(self.pos_step+1)*(batch_size / 2)]
        idx_neg = self.neg_idx[self.neg_step*(batch_size / 2):(self.neg_step+1)*(batch_size / 2)]

        if len(idx_pos) < (batch_size / 2):
            self.pos_step = 0
            idx_pos = self.pos_idx[:(batch_size / 2)]

        if len(idx_neg) < batch_size / 2:
            self.neg_step = 0
            idx_neg = self.neg_idx[:(batch_size / 2)]


        x = np.concatenate([self.x[idx_pos], self.x[idx_neg]])
        y = np.concatenate([self.y[idx_pos], self.y[idx_neg]])
        p = np.random.permutation(len(x))

        self.pos_step = self.pos_step + 1 if ((self.pos_step + 1) * (batch_size / 2)) <= len(self.pos_idx) else 0
        self.neg_step = self.neg_step + 1 if ((self.neg_step + 1) * (batch_size / 2)) <= len(self.neg_idx) else 0

        return x[p], y[p]

    def set_balanced_values(self):
        self.pos_idx = [i for i, elem in enumerate(self.y) if np.argmax(elem) == 1]
        self.neg_idx = [i for i, elem in enumerate(self.y) if np.argmax(elem) == 0]

        self.pos_step = 0
        self.neg_step = 0

    def shape(self):
        return self.x.shape, self.y.shape
