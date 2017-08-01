import os
import numpy as np
from basedata import BaseData, Data


class DataPatches(BaseData):
    def __init__(self, small=True):
        self.name = 'fp-reduction-patches'
        self.small = small
        BaseData.__init__(self)

    def load(self):
        self.datadir = '/home/marysia/data/thesis/patches'
        prefix = 'small_' if self.small else ''
        extracted = os.path.exists(os.path.join(self.datadir, prefix + 'negative_test_patches.npz'))
        if extracted:
            train_pos = np.load(os.path.join(self.datadir, prefix + 'positive_train_patches.npz'))['data']
            train_neg = np.load(os.path.join(self.datadir, prefix + 'negative_train_patches.npz'))['data']
            labels = np.concatenate([np.ones(train_pos.shape[0]), np.zeros(train_neg.shape[0])])
            data = np.concatenate([train_pos, train_neg])
            data = self.preprocess(data)
            data, labels = self.shuffle(data, labels)
            self.train = Data(scope='train', x=data, y=self._one_hot_encoding(labels, self.nb_classes))

            test_pos = np.load(os.path.join(self.datadir, prefix + 'positive_test_patches.npz'))['data']
            test_neg = np.load(os.path.join(self.datadir, prefix + 'negative_test_patches.npz'))['data']
            labels = np.concatenate([np.ones(test_pos.shape[0]), np.zeros(test_neg.shape[0])])
            data = np.concatenate([test_pos, test_neg])
            data = self.preprocess(data)
            data, labels = self.shuffle(data, labels)
            self.test = Data(scope='test', x=data, y=self._one_hot_encoding(labels, self.nb_classes))

            self.val = Data(scope='val-empty', x=np.array([]), y=np.array([]))
        else:
            print(
            '%s does not seem to contain the .npz file required for loading the %s data. Please execute '
            'read_in_zipped_pickles_script.py with python3 to create .npz files.' % (
            self.datadir, self.name))

    def preprocess(self, data):
        #data = data[:, 3:10, 45:75, 45:75]
        data = data[:, 3:11, 40:70, 40:70]

        if data.dtype == np.uint8:
            data = (data - np.float32(127.5)) * np.float32(1 / 127.5)

        elif data.dtype == np.int16:
            lower = -1000
            upper = 300
            a = np.maximum(data - lower, 0)
            upper = upper - lower
            lower = 0
            a[a > upper] = upper
            a = a * (255. / (upper - lower))
            a = a.astype(np.uint8)
            data = a

        # elif data.dtype == np.int16:
        #     start_unit = -1000
        #     end_unit = 300
        #
        #
        #     data = 2 * (data.astype(np.float32) - start_unit) / (end_unit - start_unit) - 1

        else:
            print('Unsupported datatype for preprocessing.')
        s = data.shape
        data = data.reshape((s[0], s[1], s[2], s[3], 1))
        return data

    def _set_classes(self):
        self.nb_classes = 2  # positive or negative
