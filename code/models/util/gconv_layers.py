import numpy as np
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
from groupy.gconv.tensorflow_gconv.splitgconv3d import gconv3d, gconv3d_util

from base_layers import batch_normalization, activation, weight_variable, bias_variable


def _channels(x, in_group, out_group, in_channels, out_channels):
    mapping = {'Z2': in_channels * in_channels,
               'Z3': in_channels * in_channels,
               'C4': 4, 'D4': 8, 'O': 24, 'B': 8, 'BR': 16}
    in_c = int(in_channels / round(np.sqrt(mapping[in_group])))
    out_c = int(out_channels / round(np.sqrt(mapping[out_group])))
    return in_c, out_c


def gconv_wrapper2d(x, in_group, out_group, ksize=3, in_channels=None, out_channels=16):
    """
    A wrapper to perform the 2d group convolution

    Args:
        x: tensor
        in_group: str, either Z2, C4 of D4
        out_group: either C4 or D4
        ksize:
        out_channels:

    Returns:
        gconv: tensor with gconvolution performed onto it

    """

    in_channels = in_channels if in_channels is not None else out_channels

    # in channel and out channel are dependent on the group. Divide by square root of the number of elements in a group
    # to keep the same amount of parameters
    in_c, out_c = _channels(x, in_group, out_group, in_channels, out_channels)


    # utilize gconv2d_util to get gconv_indices, shape info and w_shape
    indices, shape_info, w_shape = gconv2d_util(
        h_input=in_group, h_output=out_group,
        in_channels=in_c, out_channels=out_c, ksize=ksize
    )

    # define weight variable based on w_shape
    w = weight_variable(w_shape)
    # peform the convolution operation
    gconv = gconv2d(input=x, filter=w, strides=[1, 1, 1, 1],
                    padding='SAME', gconv_indices=indices,
                    gconv_shape_info=shape_info)

    b = bias_variable([gconv.get_shape()[-1]])
    return gconv + b


def gconv_wrapper3d(x, in_group, out_group, ksize=3, in_channels=None, out_channels=16):
    in_channels = in_channels if in_channels is not None else out_channels
    in_c, out_c = _channels(x, in_group, out_group, in_channels, out_channels)

    indices, shape_info, w_shape = gconv3d_util(
        h_input=in_group, h_output=out_group,
        in_channels=in_c, out_channels=out_c, ksize=ksize
    )
    w = weight_variable(w_shape)

    gconv = gconv3d(input=x, filter=w, strides=[1, 1, 1, 1, 1],
                    padding="SAME", gconv_indices=indices, gconv_shape_info=shape_info)
    b = bias_variable([gconv.get_shape()[-1]])
    return gconv + b


def gconv_bn_act(x, in_group, out_group, ksize=3, in_channels=None, out_channels=16):
    """ 2d group convolution - batch normalization - activation """
    gconv = gconv_wrapper2d(x, in_group, out_group, ksize, in_channels, out_channels)
    bn = batch_normalization(gconv)
    act = activation(bn, key='relu')
    return act


def bn_act_gconv(x, in_group, out_group, ksize=3, in_channels=None, out_channels=16):
    """ batch normalization - activation - 2d group convolution """
    bn = batch_normalization(x)
    act = activation(bn, key='relu')
    gconv = gconv_wrapper2d(act, in_group, out_group, ksize, in_channels, out_channels)
    return gconv


def act_bn_gconv(x, in_group, out_group, ksize=3, in_channels=None, out_channels=16):
    """ activation - batch normalization - 2d group convolution """
    act = activation(x, key='relu')
    bn = batch_normalization(act)
    gconv = gconv_wrapper2d(bn, in_group, out_group, ksize, in_channels, out_channels)
    return gconv


def gconv3d_bn_act(x, in_group, out_group, ksize=3, in_channels=None, out_channels=16):
    """ 3d group convolution - batch normalization - activation function """
    gconv = gconv_wrapper3d(x, in_group, out_group, ksize, in_channels, out_channels)
    bn = batch_normalization(gconv)
    act = activation(bn)
    return act


def bn_act_gconv3d(x, in_group, out_group, ksize=3, in_channels=None, out_channels=16):
    """ batch normalization - activation function - 3d group convolution """
    bn = batch_normalization(x)
    act = activation(bn)
    gconv = gconv_wrapper3d(act, in_group, out_group, ksize, in_channels, out_channels)
    return gconv
