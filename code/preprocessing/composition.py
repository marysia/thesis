import os
import cv2
import numpy as np

from basedata import BaseData, Data

import sys
sys.path.append('..')
from utils.config import DATADIR, RESULTSDIR

class DataComposition(BaseData):
    def __init__(self, args):
        self.name = 'nodule-classification'
        self.shape = [int(dim) for dim in args.shape]

        self.train_dataset = 'lidc'
        self.val_dataset = args.val
        self.test_dataset = 'lidc'

        self.train_mean = None
        self.train_std = None

        BaseData.__init__(self)

    def load(self):
        loaded = np.load(os.path.join(DATADIR, 'patches', 'lidc-localization-patches', 'composition.npz'))



        self.solids = loaded['solid']
        self.parts = loaded['partsolid']
        self.ggo = loaded['ggo']

    def set_two_class_fold(self, fold, type):
        self.nb_classes = 2

        test_idx = list(range(fold * 10, (fold+1) * 10))
        train_idx = [i for i in xrange(50) if i not in test_idx]

        train_data = np.concatenate([self.solids[train_idx], self.parts[train_idx], self.ggo[train_idx]])
        test_data = np.concatenate([self.solids[test_idx], self.parts[test_idx], self.ggo[test_idx]])

        if type == 'solid':
            train_labels = np.array(([1] * 2 * len(train_idx)) + ([0] * len(train_idx)))
            test_labels = np.array(([1] * 20) + ([0] * 10))
        elif type == 'ggo':
            train_labels = np.array(([0] * len(train_idx)) + ([1] * 2 * len(train_idx)))
            test_labels = np.array(([0] * 10) + ([1] * 20))


        train_data = self.preprocess(train_data, 'train')
        test_data = self.preprocess(test_data, 'test')

        self.train =  Data(scope='train-'+str(fold+1), x=train_data, y=train_labels, nb_classes=2, balanced=False)
        self.test = Data(scope='test-' + str(fold+1), x=test_data, y=test_labels, nb_classes=2, balanced=False)
        self.val = Data(scope='val-empty', x=np.array([]), y=np.array([]), nb_classes=2, balanced=None)

    def set_fold(self, fold, type):
        """ fold = 0 - 5 """
        if type == '3-class':
            # get test data
            test_idx = list(range(fold * 10, (fold+1) * 10))
            test_data = np.concatenate([self.solids[test_idx], self.parts[test_idx], self.ggo[test_idx]])
            test_labels = np.array(([2] * 10) + ([1] * 10) + ([0] * 10))


            train_idx = [i for i in xrange(50) if i not in test_idx]

            if self.val_dataset != 'empty':
                val_idx = train_idx[0:5]
                train_idx = train_idx[5:]

            train_data = np.concatenate([self.solids[train_idx], self.parts[train_idx], self.ggo[train_idx]])
            train_labels = np.array(([2] * len(train_idx)) + ([1] * len(train_idx)) + ([0] * len(train_idx)))

            train_data = self.preprocess(train_data, 'train')
            test_data = self.preprocess(test_data, 'test')

            self.train =  Data(scope='train-'+str(fold+1), x=train_data, y=train_labels, nb_classes=3, balanced=False)
            self.test = Data(scope='test-' + str(fold + 1), x=test_data, y=test_labels, nb_classes=3, balanced=False)

            if self.val_dataset != 'empty':
                val_data = np.concatenate([self.solids[val_idx], self.parts[val_idx], self.ggo[val_idx]])
                val_labels =  np.array(([2] * len(val_idx)) + ([1] * len(val_idx)) + ([0] * len(val_idx)))
                val_data = self.preprocess(val_data, 'val')
                self.val = Data(scope='val-'+str(fold+1), x=val_data, y=val_labels, nb_classes=3, balanced=False)
            else:
                self.val = Data(scope='val-empty', x=np.array([]), y=np.array([]), nb_classes=3, balanced=None)
        else:
            self.set_two_class_fold(fold, type)


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

        data[data < 0] = 0.0
        data[data > 1.0] = 1.0
        if scope != 'train':
            # reshape
            data = self._data_reshape(data)

        # subtract train mean and divide by train std
        if scope == 'train':
            self.mean = np.mean(data)
            data -= self.mean
            self.std = np.std(data)
            data /= self.std
        else:
            data -= self.mean
            data /= self.std
        #data = np.around(data, decimals=1)

        # reshape for channel
        s = data.shape
        if len(data.shape) == 4:
            data = data.reshape((s[0], s[1], s[2], s[3], 1))
        else:
            data = data.reshape((s[0], s[1], s[2], 1))

        return data

    def _set_classes(self):
        self.nb_classes = 3  # ground-glass, part-solid or solid
