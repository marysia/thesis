import os
import numpy as np
from basedata import BaseData, Data


class DataPatches(BaseData):
    def __init__(self, small=True, shape=(8, 30, 30)):
        self.name = 'fp-reduction-patches'
        self.small = small
        self.shape = shape
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

    def _data_reshape(self, data):
        data_offset = [int(size/2) for size in data.shape[1:]]
        data_diff = [int(size/2) for size in self.shape]
        data_diff_min = data_diff
        data_diff_max = []
        for i, elem in enumerate(data_diff):
            if self.shape[i] % 2 == 0:
                data_diff_max.append(elem)
            else:
                data_diff_max.append(elem+1)
        data = data[:, (data_offset[0] - data_diff_min[0]):(data_offset[0] + data_diff_max[0]),
                    (data_offset[1] - data_diff_min[1]):(data_offset[1] + data_diff_max[1]),
                    (data_offset[2] - data_diff_min[2]):(data_offset[2] + data_diff_max[2])]

        if data.shape[1] == 1:
            data = data.reshape(data.shape[0], data.shape[2], data.shape[3])

        print('Data shape: ', data.shape)
        print('Data desired shape: ', self.shape)
        return data


    def preprocess(self, data):
        #data = data[:, 3:10, 45:75, 45:75]
        #data = data[:, 3:11, 40:70, 40:70]
        data = self._data_reshape(data)
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
        if len(data.shape) == 4:
            data = data.reshape((s[0], s[1], s[2], s[3], 1))
        else:
            data = data.reshape((s[0], s[1], s[2], 1))
        return data

    def _set_classes(self):
        self.nb_classes = 2  # positive or negative
