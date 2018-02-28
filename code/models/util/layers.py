import tensorflow as tf

import base_layers as base

def conv3d_bn_act(tensor, filter_shape=[3, 3, 3], nb_channels_out=16, activation='crelu'):
    '''
    3D convolution - batch normalization - activation
    '''
    tensor = base.convolution3d(tensor, filter_shape, nb_channels_out)
    tensor = base.batch_normalization(tensor)
    tensor = base.activation(tensor, key=activation)
    return tensor


def bn_act_conv3d(tensor, filter_shape=[3, 3, 3], nb_channels_out=16, activation='relu'):
    '''
    Batch normalization - activation - 3D convolution
    '''
    tensor = base.batch_normalization(tensor)
    tensor = base.activation(tensor, key=activation)
    tensor = base.convolution3d(tensor, filter_shape, nb_channels_out)
    return tensor
