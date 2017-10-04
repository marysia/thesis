import tensorflow as tf

import base_layers as base


def bn_act_conv2d(tensor, filter_shape=[3, 3], nb_channels_out=16):
    '''
    Batch Normalization -> Relu activation -> 2D convolution
    '''
    tensor = base.batch_normalization(tensor)
    tensor = base.activation(tensor, key='relu')
    tensor = base.convolution2d(tensor, filter_shape, nb_channels_out)
    return tensor


def conv2d_bn_act(tensor, filter_shape=[3, 3], nb_channels_out=16, activation='relu'):
    ''' 2D convolution -> relu activation -> batch normalization '''
    tensor = base.convolution2d(tensor, filter_shape, nb_channels_out)
    tensor = base.activation(tensor, key=activation)
    tensor = base.batch_normalization(tensor)
    return tensor


def act_bn_conv2d(tensor, filter_shape=[3, 3], nb_channels_out=16, activation='relu'):
    tensor = base.activation(tensor, key=activation)
    tensor = base.batch_normalization(tensor)
    tensor = base.convolution2d(tensor, filter_shape=filter_shape, nb_channels_out=nb_channels_out)
    return tensor


def conv_act_pool(tensor, filter_shape, nb_channels_out):
    tensor = base.convolution2d(tensor, filter_shape=filter_shape, nb_channels_out=nb_channels_out)
    tensor = base.activation(tensor, key='relu')
    tensor = tf.nn.max_pool(tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tensor


def conv3d_bn_act(tensor, filter_shape=[3, 3, 3], nb_channels_out=16, activation='crelu'):
    tensor = base.convolution3d(tensor, filter_shape, nb_channels_out)
    tensor = base.batch_normalization(tensor)
    tensor = base.activation(tensor, key=activation)
    return tensor


def bn_act_conv3d(tensor, filter_shape=[3, 3, 3], nb_channels_out=16, activation='relu'):
    tensor = base.batch_normalization(tensor)
    tensor = base.activation(tensor, key=activation)
    tensor = base.convolution3d(tensor, filter_shape, nb_channels_out)
    return tensor
