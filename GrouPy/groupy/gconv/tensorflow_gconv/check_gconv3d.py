
import numpy as np
import tensorflow as tf

from groupy.gconv.tensorflow_gconv.splitgconv3d import gconv3d_util, gconv3d

from groupy.gfunc.z3func_array import Z3FuncArray
from groupy.gfunc.otfunc_array import OtFuncArray
import groupy.garray.O_array as O

def check_o_z3_conv_equivariance():
    li = [3, 5, 7, 9]
    for ksize in li:
        try:
            im = np.random.randn(2, ksize, ksize, ksize, 1)
            x, y = make_graph('Z3', 'O', ksize)
            check_equivariance(im, x, y, Z3FuncArray, OtFuncArray, O)
        except:
            print('O - Z3: Fails for ksize=', ksize)
# works for ksize is odd and > 3
def check_o_o_conv_equivariance():
    li = [3, 5, 7, 9]
    for ksize in li:
        try:
            im = np.random.randn(2, ksize, ksize, ksize, 24)
            x, y = make_graph('O', 'O', ksize)
            check_equivariance(im, x, y, OtFuncArray, OtFuncArray, O)
        except:
            print('O - O: Fails for ksize=', ksize)

def make_graph(h_input, h_output, ksize):
    gconv_indices, gconv_shape_info, w_shape = gconv3d_util(
        h_input=h_input, h_output=h_output, in_channels=1, out_channels=1, ksize=ksize)
    nti = gconv_shape_info[-2]

    x = tf.placeholder(tf.float32, [None, ksize, ksize, ksize, 1 * nti])
    w = tf.Variable(tf.truncated_normal(w_shape, stddev=1.))

    y = gconv3d(input=x, filter=w, strides=[1, 1, 1, 1, 1], padding='SAME',
                gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)
    return x, y


def check_equivariance(im, input, output, input_array, output_array, point_group):

    # Transform the image
    f = input_array(im.transpose((0, 4, 1, 2, 3)))
    g = point_group.rand()
    gf = g * f
    im1 = gf.v.transpose((0, 2, 3, 4, 1))

    # Compute
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    yx = sess.run(output, feed_dict={input: im})
    yrx = sess.run(output, feed_dict={input: im1})
    sess.close()

    # Transform the computed feature maps
    fmap1_garray = output_array(yrx.transpose((0, 4, 1, 2, 3)))
    r_fmap1_data = (g.inv() * fmap1_garray).v.transpose((0, 2, 3, 4, 1))

    diff = np.abs(yx - r_fmap1_data)
    print '\nSum of differences: ', diff.sum()
    print 'Median * total: ', np.median(diff) * yx.size
    print 'Max: ', np.max(diff), ' Mean: ', np.mean(diff), ' Median: ', np.median(diff)
    print 'Size of yx: ', yx.size

    assert np.allclose(yx, r_fmap1_data, rtol=1e-5, atol=1e-3)
