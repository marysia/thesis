import numpy as np
import skimage.transform as tf

def rotate_transform_batch(x, rotation=None):
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

