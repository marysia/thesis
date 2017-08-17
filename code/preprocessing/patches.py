import os
import numpy as np
from basedata import BaseData, Data


class DataPatches(BaseData):
    def __init__(self, small=True, shape=(8, 30, 30), train_balanced=True, test_balanced=False,
                 train='nlst-balanced', val='nlst-unbalanced', test='lidc-localization'):
        self.name = 'fp-reduction-patches'
        self.shape = shape

        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test

        BaseData.__init__(self)

    def load(self):
        self.train = self.get_dataset(self.train_dataset, 'train')
        self.val = self.get_dataset(self.val_dataset, 'val')
        self.test = self.get_dataset(self.test_dataset, 'test')

    def get_dataset(self, dataset, scope):
        if dataset == None:
            return Data(scope='%s-empty' % scope, x=np.array([]), y=np.array([]), id=np.array([]), balanced=None)

        if dataset == 'nlst-balanced':
            return self.nlst(scope, True)

        if dataset == 'nlst-unbalanced':
            return self.nlst(scope, False)

        if dataset == 'lidc-localization':
            return self.lidc_localization(scope, False)

    def nlst(self, scope, balanced):
        f_scope = 'train' if scope == 'train' else 'test'

        datadir = '/home/marysia/data/thesis/patches/nlst-patches/'
        prefix = 'all_' if not balanced else ''

        pos = np.load(os.path.join(datadir, 'positive_%s_patches.npz' % (f_scope)))['data']
        neg = np.load(os.path.join(datadir, '%snegative_%s_patches.npz' % (prefix, f_scope)))['data']
        labels = np.concatenate([np.ones(pos.shape[0]), np.zeros(neg.shape[0])])
        data = np.concatenate([pos, neg])
        data = self.preprocess(data)

        data, labels, p = self.shuffle(data, labels)
        return Data(scope=scope, x=data, y=self._one_hot_encoding(labels, self.nb_classes), id=p, balanced=balanced)

    def lidc_localization(self, scope, balanced):
        datadir = '/home/marysia/data/thesis/patches/lidc-localization-patches/'

        pos = np.load(os.path.join(datadir, 'positive_patches.npz'))['data']
        neg = np.load(os.path.join(datadir, 'negative_patches.npz'))['data']


        labels = np.concatenate([np.ones(pos.shape[0]), np.zeros(neg.shape[0])])
        data = np.concatenate([pos, neg])
        data = self.preprocess(data)

        data, labels, p = self.shuffle(data, labels)
        return Data(scope=scope, x=data, y=self._one_hot_encoding(labels, self.nb_classes), id=p, balanced=balanced)

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
