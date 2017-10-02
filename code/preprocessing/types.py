import os

import numpy as np

from basedata import BaseData, Data, Data2

class DataTypes(BaseData):
    def __init__(self, args):
        self.name = 'nodule-classification'
        self.shape = [int(dim) for dim in args.shape]

        self.train_dataset = args.train
        self.val_dataset = args.val
        self.test_dataset = args.test

        self.train_mean = None
        self.train_std = None

        BaseData.__init__(self)

    def load(self):

        self.train = self.get_dataset(self.train_dataset, 'train')
        self.val = self.get_dataset(self.val_dataset, 'val')
        self.test = self.get_dataset(self.test_dataset, 'test')

    def get_dataset(self, dataset, scope):

        if dataset == 'empty':
            return Data(scope='%s-empty' % scope, x=np.array([]), y=np.array([]), id=np.array([]), balanced=None)

        if dataset == 'nlst':
            return self.nlst(scope)
        if dataset == 'lidc':
            return self.lidc(scope)

    def nlst(self, scope):
        datadir = '/home/marysia/data/thesis/patches/nlst-patches/'
        f_scope = 'train' if scope == 'train' else 'test'

        loaded = np.load(os.path.join(datadir, 'positive_%s_patches_meta.npz' % f_scope))
        data = loaded['data']
        data = self.preprocess(data, scope)
        meta = loaded['meta']
        types = np.array([elem['type'] for elem in meta])

        index = [i for i, value in enumerate(types) if value == 'fluid' or value == 'other' or value == 'unknown']
        data = np.delete(data, index)
        types = np.delete(types, index)
        # transform types to 0, 1, 2, etc.
        types[types == 'soft-tissue'] = 0
        types[types == 'mixed'] = 1
        types[types == 'ground-glass'] = 2

        return Data2(scope=scope, x=data, y=types, nb_classes=3, balanced=True)



    def _data_reshape(self, data):
        """
        Resize the data to a smaller patch, e.g. 13x120x120 to 8x30x30, by determining the center point and cutting
        around it.
        """
        data_offset = [int(size / 2) for size in data.shape[1:]]
        data_diff = [int(size / 2) for size in self.shape]
        data_diff_min = data_diff
        data_diff_max = []
        for i, elem in enumerate(data_diff):
            if self.shape[i] % 2 == 0:
                data_diff_max.append(elem)
            else:
                data_diff_max.append(elem + 1)
        data = data[:, (data_offset[0] - data_diff_min[0]):(data_offset[0] + data_diff_max[0]),
               (data_offset[1] - data_diff_min[1]):(data_offset[1] + data_diff_max[1]),
               (data_offset[2] - data_diff_min[2]):(data_offset[2] + data_diff_max[2])]

        if data.shape[1] == 1:
            data = data.reshape(data.shape[0], data.shape[2], data.shape[3])
        return data


    def preprocess(self, data, scope):
        """
        Preprocess the data by reshaping it to the target shape, normalizing it between values of [-1, 1],
        subtracting the train mean and dividing by the std, and reshaping it to contain the channel.
        """
        if scope != 'train':
            # reshape
            data = self._data_reshape(data)

        # normalize
        if data.dtype == np.int16:
            start_unit = -1000
            end_unit = 300
            data = 2 * (data.astype(np.float32) - start_unit) / (end_unit - start_unit) - 1

        # subtract train mean and divide by train std
        if scope == 'train':
            self.mean = np.mean(data)
            data -= self.mean
            self.std = np.std(data)
            data /= self.std
        else:
            data -= self.mean
            data /= self.std

        # reshape for channel
        s = data.shape
        if len(data.shape) == 4:
            data = data.reshape((s[0], s[1], s[2], s[3], 1))
        else:
            data = data.reshape((s[0], s[1], s[2], 1))
        return data

    def _set_classes(self):
        self.nb_classes = 3  # ground-glass, part-solid or solid
