import warnings

import numpy as np
import scipy.ndimage

def add_noise(x):
    """ Add noise """
    x += np.random.randn(x.shape[0], x.shape[1], x.shape[2]) * 0.05
    return x

def add_blur(x, scalar):
    return scipy.ndimage.gaussian_filter(x, sigma=scalar)

def rotate_dataset(x, rotation):
    """ Rotates batch n times 90 degrees. """
    x_out = np.empty_like(x)
    for i in xrange(x.shape[0]):
        for j in xrange(x.shape[1]):
            im = x[i, j, :, :, 0]
            im = np.rot90(im, rotation)
            x_out[i, j, :, :, 0] = im
    return x


def flip_dataset(x, flip):
    """ Flips the entire batch in the z, y or x axis."""
    if flip == 0:
        return x
    if flip == 1:
        return x[:, ::-1, :, :, :]
    if flip == 2:
        return x[:, :, ::-1, :, :]
    if flip == 3:
        return x[:, :, :, ::-1, :]

def scale_volume(x, scalar):
    """ Scales the volume between 0.9 and 1.1 time the original volume. """
    return scipy.ndimage.zoom(x, scalar)


def rotate_volume(x, rotation):
    """ Rotates the volume 0-360 degrees. """
    x_out = np.empty_like(x)
    for i in xrange(x.shape[0]):
        im = x[i, :, :]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            im = scipy.ndimage.rotate(im, rotation, reshape=False)

        x_out[i, :, :] = im
    return x_out

def flip_volume(x, flip):
    """ Flips the volume in the z, y or x axis. """
    if flip == 0:
        return x
    if flip == 1:
        return x[::-1, :, :]
    if flip == 2:
        return x[:, ::-1, :]
    if flip == 3:
        return x[:, :, ::-1]

def crop_volume(x, shape):
    """ Crops the volume to the desired shape. """
    center = {
        'x': x.shape[2] / 2 + np.random.randint(-3, 4),
        'y': x.shape[1] / 2 + np.random.randint(-3, 4),
        'z': x.shape[0] / 2 + np.random.randint(0, 1)  # max of 1 translation
    }

    dif = {
        'x': shape[2] / 2,
        'y': shape[1] / 2,
        'z': shape[0] / 2
    }

    return x[center['z'] - dif['z']:center['z'] + dif['z'], center['y'] - dif['y']:center['y'] + dif['y'],
           center['x'] - dif['x']:center['x'] + dif['x']]

def augment_batch(x, transformations, shape, keep_prob=.2):
    """  Perform augmentations on batch with chance.
    Possible augmentations: flips, rotations, scaling. Always crop, to get the right shape.
    """

    x_out = np.zeros((x.shape[0], shape[0], shape[1], shape[2], 1))
    for i in xrange(x.shape[0]):
        flip = np.random.randint(0, 4)
        rotation = np.random.randint(0, 360)
        scale = np.random.uniform(low=.8, high=1.2)
        blur = np.random.randint(0, 2)

        volume = x[i, :, :, :, 0]
        volume = scale_volume(volume, scale) if 'scale' in transformations and np.random.random() > keep_prob else volume
        volume = rotate_volume(volume, rotation) if 'rotate' in transformations and np.random.random() > keep_prob else volume
        volume = flip_volume(volume, flip) if 'flip' in transformations and np.random.random() > keep_prob else volume
        volume = add_noise(volume) if 'noise' in transformations and np.random.random() > keep_prob else volume
        volume = add_blur(volume, blur) if 'blur' in transformations and np.random.random() > keep_prob else volume
        # crop with translation (calculate after scaling/rotation)
        volume = crop_volume(volume, shape)
        x_out[i, :, :, :, 0] = volume

    return x_out


def augment_dataset(x, rotations, flip):
    """ Augment the dataset to average over all symmetries. No cropping necessary,
    as test set is already the correct shape."""

    x = rotate_dataset(x, rotations)
    x = flip_dataset(x, flip)
    return x
