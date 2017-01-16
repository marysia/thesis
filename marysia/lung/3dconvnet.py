# imports
import numpy as np
import os

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

datadir = '/home/marysia/thesis/data/LUNG/'
positives_data = np.load(datadir+'positives.npz')['patch']
positives_labels = np.ones(len(positives_data))

negatives_data = np.load(datadir+'negatives.npz')['patch']
negatives_labels = np.zeros(len(negatives_data))

X = np.concatenate([positives_data, negatives_data])
Y = np.append(positives_labels, negatives_labels)
X = X.reshape([-1, 7, 72, 72, 1])
Y = to_categorical(Y, 2)
X, Y = shuffle(X, Y)

train, val = (2000, 2100)
X_train = X[:train, :, :, :]
X_val = X[train:val, :, :, :]
X_test = X[val:, :, :, :]

Y_train = Y[:train, :]
Y_val = Y[train:val, :]
Y_test = Y[val:, :]

print(Y.shape)
print(X.shape)

# Convolutional network building
network = input_data(shape=[None, 7, 72, 72, 1])
network = conv_3d(network, 32, 3, activation='relu')
network = max_pool_3d(network, 2)
network = conv_3d(network, 64, 3, activation='relu')
network = conv_3d(network, 64, 3, activation='relu')
network = max_pool_3d(network, 2)
#network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
print('Network created.')

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X_train, Y_train, n_epoch=5, shuffle=True, validation_set=(X_val, Y_val),
          show_metric=True, batch_size=96, run_id='lung_cnn')

text_file = open("output_small.txt", "w")
text_file.write('\nScore: %.6f \n' % model.evaluate(X_test,Y_test)[0])
text_file.close()




