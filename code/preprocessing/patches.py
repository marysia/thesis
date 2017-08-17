import os
import numpy as np
from basedata import BaseData, Data, UnbalancedData


class DataPatches(BaseData):
    def __init__(self, small=True, shape=(8, 30, 30), train_balanced=True, test_balanced=False):
        self.name = 'fp-reduction-patches'
        self.small = small
        self.shape = shape
        self.train_balanced = train_balanced
        self.test_balanced = test_balanced
        BaseData.__init__(self)

    def load(self):
        self.datadir = '/home/marysia/data/thesis/patches'
        prefix = 'small_' if self.small else ''
        self.val = Data(scope='val-empty', x=np.array([]), y=np.array([]))

        # set train set
        if self.train_balanced:
            train_pos = np.load(os.path.join(self.datadir, prefix + 'positive_train_patches.npz'))['data']
            train_neg = np.load(os.path.join(self.datadir, prefix + 'negative_train_patches.npz'))['data']
            labels = np.concatenate([np.ones(train_pos.shape[0]), np.zeros(train_neg.shape[0])])
            data = np.concatenate([train_pos, train_neg])
            data = self.preprocess(data)

            data, labels, p = self.shuffle(data, labels)
            self.id_train = p
            self.train = Data(scope='train', x=data, y=self._one_hot_encoding(labels, self.nb_classes))
        else:
            train_pos = np.load(os.path.join(self.datadir, prefix + 'positive_train_patches.npz'))['data']
            train_neg = np.load(os.path.join(self.datadir, prefix + 'negative_all_train_patches.npz'))['data']

            train_pos = self.preprocess(train_pos)
            train_neg = self.preprocess(train_neg)
            y_pos = self._one_hot_encoding(np.ones(train_pos.shape[0]), self.nb_classes)
            y_neg = self._one_hot_encoding(np.zeros(train_neg.shape[0]), self.nb_classes)
            self.train = UnbalancedData(scope='train-pos', x_pos=train_pos, y_pos=y_pos, x_neg=train_neg, y_neg=y_neg)

        # set test set
        if self.test_balanced:
            test_pos = np.load(os.path.join(self.datadir, prefix + 'positive_test_patches.npz'))['data']
            test_neg = np.load(os.path.join(self.datadir, prefix + 'negative_test_patches.npz'))['data']
            labels = np.concatenate([np.ones(test_pos.shape[0]), np.zeros(test_neg.shape[0])])
            data = np.concatenate([test_pos, test_neg])
            data = self.preprocess(data)
            data, labels, p = self.shuffle(data, labels)
            self.id_test = p  # offset: maximum value of idx train + 1

            self.test = Data(scope='test', x=data, y=self._one_hot_encoding(labels, self.nb_classes))
        else:
            test_pos = np.load(os.path.join(self.datadir, prefix + 'all_positive_test_patches.npz'))['data']
            test_neg = np.load(os.path.join(self.datadir, prefix + 'all_negative_all_test_patches.npz'))['data']
            labels = np.concatenate([np.ones(test_pos.shape[0]), np.zeros(test_neg.shape[0])])
            data = np.concatenate([test_pos, test_neg])
            data = self.preprocess(data)
            data, labels, p = self.shuffle(data, labels)
            self.id_test = p
            self.test = Data(scope='test', x=data, y=self._one_hot_encoding(labels, self.nb_classes))



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
        return data


    def preprocess(self, data):
        #data = data[:, 3:10, 45:75, 45:75]
        #data = data[:, 3:11, 40:70, 40:70]
        data = self._data_reshape(data)
        if data.dtype == np.uint8:
            data = (data - np.float32(127.5)) * np.float32(1 / 127.5)

        # elif data.dtype == np.int16:
        #     lower = -1000
        #     upper = 300
        #     data = np.maximum(data - lower, 0)
        #     upper = upper - lower
        #     lower = 0
        #     data[data > upper] = upper
        #     data = data * (2. / (upper - lower)) - 1
        #     data = data.astype(np.float32)
        # #
        if data.dtype == np.int16:
            start_unit = -1000
            end_unit = 300


            data = 2 * (data.astype(np.float32) - start_unit) / (end_unit - start_unit) - 1

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
