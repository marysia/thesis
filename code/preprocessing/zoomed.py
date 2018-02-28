import warnings

import numpy as np
import scipy.ndimage

target_mm = (1, 1, 1)
current_mm = (1.25, .5, .5)
zoom_factor = (np.array(current_mm) / np.array(target_mm))

orig_folder = '/home/marysia/data/thesis/patches/'
new_folder = '/home/marysia/data/thesis/zoomed-patches/'

scope = 'lidc-localization-patches/'

print("Starting.")
file = '/home/marysia/data/thesis/patches/nlst-patches/positive_train_patches.npz'
data = np.load(file)['data']
print("Loaded.")
print(data.shape)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    volumes = []
    for i in xrange(data.shape[0]):
        new_volume = scipy.ndimage.interpolation.zoom(data[i], zoom=zoom_factor)
        volumes.append(new_volume)

volumes = np.asarray(volumes)
print("Volume created.")

fname = file.replace('thesis/patches', 'thesis/zoomed-patches')
np.savez(fname, data=volumes)
