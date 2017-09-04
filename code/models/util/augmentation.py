import warnings

import numpy as np
import scipy.ndimage


def apply_augmentation(x, transformations, keep_prob=.5):
    """
    Function applies all listed augmentations, with a probability of _keep_prob_ to
    keep the original batch.
    Args:
        x: batch
        transformations: list of transformations
        keep_prob: probability of not applying the transformation and keeping the original batch

    Returns:
        x: transformed batch
    """
    mapping = {
        'scale': scale_batch,
        'flip': flip_batch,
        'rotate': rotate_batch
    }

    for transformation in transformations:
        if np.random.rand() > keep_prob:
            x = mapping[transformation](x)

    return x


def scale_batch(x):
    """
    Scales batch to .8-1.2.
    Args:
        x: original batch

    Returns:
        batch: scaled batch

    """
    scalar = np.random.uniform(low=.8, high=1.2)
    batch = np.empty_like(x)
    for i in xrange(x.shape[0]):
        volume = x[i, :, :, :, 0]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            volume = scipy.ndimage.zoom(volume, scalar)
        volume = _reshape_volume(volume, x.shape[1:4])
        batch[i, :, :, :, 0] = volume
    return batch


def flip_batch(x):
    """
    Flips batch over one of the axis.
    Args:
        x: original batch
    """
    flip = np.random.randint(0, len(x.shape) - 1)
    if flip == 0:
        return x
    if flip == 1:
        return x[:, ::-1, :, :, :]
    if flip == 2:
        return x[:, :, ::-1, :, :]
    if flip == 3:
        return x[:, :, :, ::-1, :]


def rotate_batch(x):
    """
    Rotates batch with 90o rotations over one of the axis.
    Args:
        x: original batch

    Returns:
        x_out: new batch.

    """
    axis = np.random.randint(0, 1)
    rotations = np.random.randint(0, 4)
    axis = 0

    x_out = np.empty_like(x)

    for i in xrange(x.shape[0]):
        im = x[i, :, :, :, :]
        for r in xrange(rotations):
            im = _rotate_3d(im, axis)
        x_out[i, :, :, :, :] = im
    return x_out


def _reshape_volume(volume, shape):
    """
    Reshapes a new volume (e.g. 10x35x35 or 6x25x25) to original volume shape (e.g. 8x30x30)
    Args:
        volume: Volume to be reshaped
        shape: target shape

    Returns:
        result: resulting volume

    """
    differences = abs(np.array(shape) - np.array(volume.shape)) + 1
    x_offset = np.random.randint(0, differences[1])
    y_offset = np.random.randint(0, differences[2])
    z_offset = np.random.randint(0, differences[0])

    # size hasn't changed
    if volume.shape == shape:
        result = volume

    # new volume is smaller than original
    elif volume.shape < shape:
        result = np.zeros(shape)
        result[z_offset:z_offset + volume.shape[0], x_offset:x_offset + volume.shape[1],
        y_offset:y_offset + volume.shape[2]] = volume

    # new volume is bigger than original
    elif volume.shape > shape:
        result = volume[z_offset:z_offset + shape[0], x_offset:x_offset + shape[1], y_offset:y_offset + shape[2]]

    return result


def _rotate_3d(im, axis):
    im_out = np.empty_like(im)
    if axis == 0:
        for j in xrange(im.shape[0]):
            im_out[j, :, :, :] = np.rot90(im[0], 3)
    if axis == 1:
        for j in xrange(im.shape[2]):
            im_out[j, :, :, :] = np.rot90(im[:, :, :, j], 3)
    return im_out
