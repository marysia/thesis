import os
import numpy as np
from basedata import BaseData, Data

class DataMNIST(BaseData):
    def __init__(self):
        self.name = 'MNIST-rot'
        BaseData.__init__(self)
        #super(DataMNIST, self).__init__()

    def load(self):
        extracted = os.path.exists(os.path.join(self.datadir, 'mnist-rot', 'mnist_rot_train.npz'))
        if extracted:
            train = np.load('/home/marysia/thesis/data/mnist-rot/mnist_rot_train.npz')
            train_data = train['data']
            train_mean = np.mean(train['data'])
            train_data -= train_mean
            train_std = np.std(train_data)
            train_data /= train_std

            self.train = Data(scope='train', x=train_data, y=self._one_hot_encoding(train['labels'], self.nb_classes))

            val = np.load('/home/marysia/thesis/data/mnist-rot/mnist_rot_valid.npz')
            val_data = val['data']
            val_data -= train_mean
            val_data /= train_std
            self.val = Data(scope='val', x=val_data, y=self._one_hot_encoding(val['labels'], self.nb_classes))

            test = np.load('/home/marysia/thesis/data/mnist-rot/mnist_rot_test.npz')
            test_data = test['data']
            test_data -= train_mean
            test_data /= train_std
            self.test = Data(scope='test', x=test_data, y=self._one_hot_encoding(test['labels'], self.nb_classes))


        else:
            # see marysia/mnist-rot/data for data extraction
            pass

    def preprocess(self):
        pass

    def _set_classes(self):
        self.nb_classes = 10