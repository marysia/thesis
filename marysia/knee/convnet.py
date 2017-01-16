from __future__ import division, print_function, absolute_import
import os

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

# Data loading and preprocessing
import numpy as np
datadir = '/home/marysia/thesis/data/'
data = np.load(datadir+'KNEE/train.npz')
X = data['data']
Y = data['labels']
data = np.load(datadir+'KNEE/valid.npz')
X_val = data['data'] 
Y_val = data['labels']
data = np.load(datadir+'KNEE/test.npz')
X_test = data['data']
Y_test = data['labels']

Y = to_categorical(Y, 2)
Y_val = to_categorical(Y_val, 2)
Y_test = to_categorical(Y_test, 2)
print(X.shape)
print(Y.shape)

print('Loaded.')

# Convolutional network building
#network = input_data(shape=[None, 1, 100, 200],
                     #data_preprocessing=img_prep,
                     #data_augmentation=img_aug)
network = input_data(shape=[None, 1, 100, 200])
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
print('Network created.')

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=3)
model.fit(X, Y, n_epoch=5, shuffle=True, validation_set=(X_val, Y_val),
          show_metric=True, batch_size=96, run_id='knee_cnn')

print(model.evaluate(X_test, Y_test))
