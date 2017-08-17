import os
import numpy as np

from basedata import Data
from patches import DataPatches

class LIDCTestPatches(DataPatches):

    def load(self):
        self.datadir = '/home/marysia/data/thesis/lidc-patches'
        self.train = Data(scope='train-empty', x=np.array([]), y=np.array([]))
        self.val = Data(scope='val-empty', x=np.array([]), y=np.array([]))

        test_pos = np.load(os.path.join(self.datadir, 'positive_test_patches.npz'))
        test_neg = np.load(os.path.join(self.datadir, 'negative_test_patches.npz'))

        pos_data = test_pos['data']
        neg_data = test_neg['data']

        data = np.concatenate([pos_data, neg_data])
        labels = np.concatenate([np.ones(pos_data.shape[0]), np.zeros(neg_data.shape[0])])

        data = self.preprocess(data)
        data, labels, p = self.shuffle(data, labels)
        self.test = Data(scope='test', x=data, y=self._one_hot_encoding(labels, self.nb_classes))


