from __future__ import division, print_function, absolute_import

import tflearn

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

Y = tflearn.data_utils.to_categorical(Y, 2)
Y_val = tflearn.data_utils.to_categorical(Y_val, 2)
Y_test = tflearn.data_utils.to_categorical(Y_test, 2)
print(X.shape)
print(Y.shape)

# X = X.reshape(-1, 100, 200, 1)
# X_val = X_val.reshape(-1, 100, 200, 1)
# X_test = X_test.reshape(-1, 100, 200, 1)
print(X.shape)
print(X_val.shape)
print(X_test.shape)

print('Loaded.')

n = 5
# Building Residual Network
net = tflearn.input_data(shape=[None, 1, 100, 200])
                         #data_preprocessing=img_prep,
                         #data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)

# Regression
net = tflearn.fully_connected(net, 2, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_knee',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.fit(X, Y, n_epoch=10, validation_set=(X_val, Y_val),
          snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=128, shuffle=True,
          run_id='resnet_knee')

print(model.evaluate(X_test, Y_test))
