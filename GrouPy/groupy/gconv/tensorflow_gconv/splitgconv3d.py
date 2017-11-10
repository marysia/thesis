import tensorflow as tf

from groupy.gconv.make_gconv_indices import make_o_z3_indices, \
    make_o_ot_indices, make_c4h_z3_indices, make_c4h_c4ht_indices, make_d4h_z3_indices, make_d4h_d4ht_indices, make_oh_z3_indices, make_oh_oht_indices, flatten_indices_3d
from groupy.gconv.tensorflow_gconv.transform_filter import transform_filter_3d_nhwc


def gconv3d(input, filter, strides, padding, gconv_indices, gconv_shape_info,
            use_cudnn_on_gpu=None, data_format='NHWC', name=None):
    if data_format != 'NHWC':
        raise NotImplemented('Currently only NHWC data_format is supported. Got:' + str(data_format))

    # Transform the filters
    transformed_filter = transform_filter_3d_nhwc(w=filter, flat_indices=gconv_indices, shape_info=gconv_shape_info)

    # Convolve input with transformed filters
    conv = tf.nn.conv3d(input=input, filter=transformed_filter, strides=strides, padding=padding, name=name)

    return conv


def gconv3d_util(h_input, h_output, in_channels, out_channels, ksize):
    """
    Convenience function for setting up static data required for the G-Conv.
     This function returns:
      1) an array of indices used in the filter transformation step of gconv2d
      2) shape information required by gconv2d
      5) the shape of the filter tensor to be allocated and passed to gconv2d

    :param h_input: one of ('Z2', 'C4', 'D4'). Use 'Z2' for the first layer. Use 'C4' or 'D4' for later layers.
    :param h_output: one of ('C4', 'D4'). What kind of transformations to use (rotations or roto-reflections).
      The choice of h_output of one layer should equal h_input of the next layer.
    :param in_channels: the number of input channels. Note: this refers to the number of (3D) channels on the group.
    The number of 2D channels will be 1, 4, or 8 times larger, depending the value of h_input.
    :param out_channels: the number of output channels. Note: this refers to the number of (3D) channels on the group.
    The number of 2D channels will be 1, 4, or 8 times larger, depending on the value of h_output.
    :param ksize: the spatial size of the filter kernels (typically 3, 5, or 7).
    :return: gconv_indices
    """

    if h_input == 'Z3' and h_output == 'O':
        gconv_indices = flatten_indices_3d(make_o_z3_indices(ksize=ksize))
        nti = 1
        nto = 24
    elif h_input == 'O' and h_output == 'O':
        gconv_indices = flatten_indices_3d(make_o_ot_indices(ksize=ksize))
        nti = 24
        nto = 24
    elif h_input == 'Z3' and h_output == 'C4H':
        gconv_indices = flatten_indices_3d(make_c4h_z3_indices(ksize=ksize))
        nti = 1
        nto = 8
    elif h_input == 'C4H' and h_output == 'C4H':
        gconv_indices = flatten_indices_3d(make_c4h_c4ht_indices(ksize=ksize))
        nti = 8
        nto = 8
    elif h_input == 'Z3' and h_output == 'D4H':
        gconv_indices = flatten_indices_3d(make_d4h_z3_indices(ksize=ksize))
        nti = 1
        nto = 16
    elif h_input == 'D4H' and h_output == 'D4H':
        gconv_indices = flatten_indices_3d(make_d4h_d4ht_indices(ksize=ksize))
        nti = 16
        nto = 16
    elif h_input == 'Z3' and h_output == 'OH':
        gconv_indices = flatten_indices_3d(make_oh_z3_indices(ksize=ksize))
        nti = 1
        nto = 48
    elif h_input == 'OH' and h_output == 'OH':
        gconv_indices = flatten_indices_3d(make_oh_oht_indices(ksize=ksize))
        nti = 48
        nto = 48
    else:
        raise ValueError('Unknown (h_input, h_output) pair:' + str((h_input, h_output)))

    w_shape = (ksize, ksize, ksize, in_channels * nti, out_channels)
    gconv_shape_info = (out_channels, nto, in_channels, nti, ksize)
    return gconv_indices, gconv_shape_info, w_shape


def gconv2d_addbias(input, bias, nti=8):
    """
    In a G-CNN, the feature maps are interpreted as functions on a group G instead of functions on the plane Z^2.
    Just like how we use a single scalar bias per 2D feature map, in a G-CNN we should use a single scalar bias per
    G-feature map. Failing to do this breaks the equivariance and typically hurts performance.
    A G-feature map usually consists of a number (e.g. 4 or 8) adjacent channels.
    This function will add a single bias vector to a stack of feature maps that has e.g. 4 or 8 times more 2D channels
    than G-channels, by replicating the bias across adjacent groups of 2D channels.

    :param input: tensor of shape (n, h, w, ni * nti), where n is the batch dimension, (h, w) are the height and width,
     ni is the number of input G-channels, and nti is the number of transformations in H.
    :param bias: tensor of shape (ni,)
    :param nti: number of transformations, e.g. 4 for C4/p4 or 8 for D4/p4m.
    :return: input with bias added
    """
    # input = tf.reshape(input, ())
    pass  # TODO
