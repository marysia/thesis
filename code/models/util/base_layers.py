'''
Contains layers for model building.
'''
import tensorflow as tf
import numpy as np


def activation(tensor, key = 'relu'):
    '''
    Activation function. Adds bias and executes activation function based on key.
    '''
    activation_functions = {
        "relu": tf.nn.relu,
        "elu": tf.nn.elu,
        "crelu": tf.nn.crelu,
        "sigmoid": tf.sigmoid,
        "tanh": tf.tanh,
    }

    with tf.name_scope(key):
        nb_channels_in = int(tensor.get_shape()[-1])
        bias = _weights_constant(shape=[nb_channels_in], value=0., name="bias")
        tensor = tf.nn.bias_add(value=tensor, bias=bias)
        tensor = activation_functions[key](tensor)
    return tensor

def dropout(tensor, keep_prob, training):
    '''
    Performs dropout if training, returns tensor otherwise.
    '''
    with tf.name_scope('dropout'):
        #tensor = tf.cond(training,
        #                 lambda: tf.nn.dropout(tensor, keep_prob=keep_prob),
        #                lambda: tensor)

        if training:
            tensor_dropout = tf.nn.dropout(tensor, keep_prob=keep_prob)
        else:
            tensor_dropout = tensor

    return tensor_dropout

def convolution2d(tensor, filter_shape, nb_channels_out):
    '''
    Performs 2d convolution
    '''
    with tf.name_scope('convolution2d'):
        strides = [1, 1, 1, 1]
        nb_channels_in = int(tensor.get_shape()[-1])
        W = weight_variable(filter_shape+[nb_channels_in, nb_channels_out])

        W = _weights_distribution(filter_shape+[nb_channels_in, nb_channels_out], "weight_distribution", schema = "he")
        b = bias_variable([nb_channels_out])
        tensor = tf.nn.conv2d(input=tensor, filter=W, strides=strides, padding="SAME") + b
    return tensor

def convolution3d(tensor, filter_shape, nb_channels_out):
    '''
    Peform 3d convolution
    '''
    with tf.name_scope('convolution3d'):
        strides = [1, 1, 1, 1, 1]
        nb_channels_in = int(tensor.get_shape()[-1])
        W = weight_variable(filter_shape+[nb_channels_in, nb_channels_out])
        #W = _weights_distribution(filter_shape+[nb_channels_in, nb_channels_out], "weight_distribution", schema = "he")
        b = bias_variable([nb_channels_out])
        tensor = tf.nn.conv3d(input=tensor, filter=W, strides=strides, padding="SAME") + b
    return tensor

def batch_normalization(tensor):
    '''
    Performs batch normalization.
    '''
    with tf.name_scope('batch_normalization'):
        # get shape
        shape = tensor.get_shape().as_list()
        nb_channels = shape[-1]

        # define trainable variables
        beta = _weights_constant(shape=[nb_channels], value=0., name='beta')
        gamma = _weights_constant(shape=[nb_channels], value=1., name='gamma')

        # define untrainable variables
        mu = _weights_constant(shape=[nb_channels], value=0., name='mu', trainable=False)
        sigma = _weights_constant(shape=[nb_channels], value=1., name='sigma', trainable=False)

        # perform batch normalization
        tensor = tf.nn.batch_normalization(x=tensor, mean=mu, variance=sigma, offset=beta,
                                           scale=gamma, variance_epsilon=1e-3)
    return tensor

def merge(tensor_x, tensor_y, method):
    '''
    Adds or concatenates two tensors, based on given method.
    '''
    with tf.name_scope(method):
        if method == 'add':
            return tf.add(x=tensor_x, y=tensor_y)
        elif method == 'concat':
            return tf.concat(axis=len(tensor_x.get_shape()) - 1, values=[tensor_x, tensor_y])

def dense(tensor, channels):
    '''
    Dense layer. Reshape tensor to flattened tensor.
    '''
    flattened_shape = np.prod(np.array(tensor.get_shape().as_list()[1:]))
    W = weight_variable([flattened_shape, channels])
    b = bias_variable([channels])
    tensor = tf.reshape(tensor, [-1, flattened_shape])
    tensor = tf.matmul(tensor, W) + b
    return tensor

def flatten(tensor):
    shape = np.prod(np.array(tensor.get_shape().as_list()[1:]))
    tensor = tf.reshape(tensor, [-1, shape])
    return tensor

def dim_reshape(tensor, z=None):
    shape = [int(dim) for dim in list(tensor.get_shape())[1:]]
    # 2d --> 3d
    if len(shape) == 3:
        y, x, c = shape
        reshape = (-1, z, y, x, c)
        return tf.reshape(tensor, reshape)
    # 3d --> 2d
    if len(shape) == 4:
        z, y, x, c = shape
        reshape = (-1, y, x, c)
        return tf.reshape(tensor, reshape), z, c

def readout(tensor, shape):
    '''
    Readout layer.
    '''
    W = weight_variable(shape)
    b = bias_variable([shape[-1]])
    #W = _weights_distribution(shape, name='weight')
    #b = _weights_constant([shape[-1]], value=0.1, name='bias')
    tensor = tf.matmul(tensor, W) + b
    return tensor

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.05, name='weight')
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.05, shape=shape, name='bias')
  return tf.Variable(initial)

def maxpool2d(tensor, strides=[1, 2, 2, 1]):
    return tf.nn.max_pool(tensor, ksize=[1, 1, 1, 1], strides=strides, padding='SAME')

def maxpool3d(tensor, strides=[1, 1, 2, 2, 1]):
    return tf.nn.max_pool3d(tensor, ksize=[1, 1, 1, 1, 1], strides=strides, padding='SAME')

def _weights_distribution(shape, name, schema = "he"):
    '''
    Returns truncated normal variable for weights.
    '''
    initialization_schemas = {
            "xavier": lambda n: 1. / n,
            "he": lambda n: 2. / n
    }

    fan_in = reduce(lambda x, y: x * y, shape[:-1], 1)
    std = initialization_schemas[schema](fan_in)

    return tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=std), name=name)

def _weights_constant(shape, value, name, trainable = True):
    '''
    Returns constant Variable for bias.
    '''
    return tf.Variable(initial_value=tf.constant(value=value, shape=shape), name=name, trainable=trainable)