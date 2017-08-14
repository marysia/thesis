import numpy as np
import skimage.transform as tf

def rotate_transform_batch2d(x, rotation=None):
    '''
    Taco's gconv_experiments. Altered for (batch, h, w, c) instead of (batch, c, h, w)
    Altered for 3D input.
    '''
    r = np.random.uniform(-0.5, 0.5, size=x.shape[0]) * rotation

    # hack; skimage.transform wants float images to be in [-1, 1]
    factor = np.maximum(np.max(x), np.abs(np.min(x)))
    x = x / factor

    x_out = np.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[3]):
            x_out[i, :, :, j] = tf.rotate(x[i, :, :, j], r[i])

    x_out *= factor

    return x_out



def rotate_transform_batch3d(x):
    axis = np.random.randint(0, 1)
    rotations = np.random.randint(0, 4)

    x_out = np.empty_like(x)

    for i in xrange(x.shape[0]):
        im = x[i, :, :, :]
        for r in xrange(rotations):
            im = rotate_3d(im, axis)
        x_out[i, :, :, :] = im
    return x_out

def rotate_3d(im, axis):
    im_out = np.empty_like(im)
    if axis == 0:
        for j in xrange(im.shape[0]):
            im_out[j, :, :] = np.rot90(im[0], 3)
    if axis == 1:
        for j in xrange(im.shape[2]):
            im_out[j, :, :] = np.rot90(im[:, :, j], 3)
    return im_out

# def rotate_transform_batch3d(x):
#
#     axis = np.random.randint(0, 3)
#     rotations = np.random.randint(0, 4)
#
#     x_out = np.empty_like(x)
#
#     for i in range(x.shape[0]):
#         im = x[i, :, :, :]
#         for r in range(rotations):
#             im = rotate_3d(im, axis)
#         x_out[i, :, :, :] = im
#
#     return x_out
#
#
# def rotate_3d(im, axis):
#     if axis == 0:
#         return np.rot90(im[:, ::-1, :].swapaxes(0, 1)[::-1, :, :].swapaxes(0, 2), 3)
#     if axis == 1:
#         return np.rot90(im, 1)
#     if axis == 2:
#         return im.swapaxes(0, 2)[::-1, :, :]