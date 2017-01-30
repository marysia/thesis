import numpy as np
from tflearn.data_utils import shuffle
from keras.utils import np_utils

def lidc():  # loading
    datadir = '/home/marysia/thesis/data/LUNG/'
    positives_data = np.load(datadir + 'lidc_positives.npz')['patch']
    positives_labels = np.ones(len(positives_data))

    negatives_data = np.load(datadir + 'lidc_negatives.npz')['patch']
    negatives_labels = np.zeros(len(negatives_data))

    X = np.concatenate([positives_data, negatives_data])
    Y = np.append(positives_labels, negatives_labels)

    print X.shape
    print Y.shape

    # Data alterations
    X, Y = shuffle(X, Y)  # shuffle indices
    X = X.reshape([-1, 1, 7, 72, 72])  # reshape
    Y = np_utils.to_categorical(Y)

    X = (X - 127.5) / 127.5
    X = X.astype('float32')

    train = 2000
    X_train = X[:train, :, :]
    X_test = X[train:, :, :]

    Y_train = Y[:train]
    Y_test = Y[train:]

    print X_train.shape
    print Y_train.shape
    return X_train, Y_train, X_test, Y_test