import numpy as np
from groupy.gconv.tensorflow_gconv.splitgconv3d import gconv3d, gconv3d_util
from base_layers import weight_variable


def get_channels(x, in_group, out_group, out_channels):
    ''' Helper function to determine the number of channels in and out based on the input tensor
    and desired output channel for a regular CNN.'''
    mapping = {'C4': 4, 'D4': 8, 'O': 24, 'C4H': 8, 'D4H': 16, 'OH': 48}
    in_c = int(x.get_shape()[-1]) / mapping[in_group] if in_group != 'Z2' and in_group != 'Z3'  else 1
    out_c = int(out_channels / round(np.sqrt(mapping[out_group])))
    return in_c, out_c


def gconvolution3d(x, in_group, out_group, out_channels, ksize=3):
    ''' Performs 3D convolution based on in- and out-group. '''
    # get input channel and output channel
    in_c, out_c = get_channels(x, in_group, out_group, out_channels)

    # get static data required for G-conv: indices, shape information and shape of filter tensor
    indices, shape_info, w_shape = gconv3d_util(
        h_input=in_group, h_output=out_group,
        in_channels=in_c, out_channels=out_c, ksize=ksize)
    w = weight_variable(w_shape)

    # perform group convolution
    gconv = gconv3d(input=x, filter=w, strides=[1, 1, 1, 1, 1],
                    padding="SAME", gconv_indices=indices, gconv_shape_info=shape_info)
    return gconv 


