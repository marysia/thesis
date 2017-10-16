import numpy as np
import cv2
def array_to_png(array, fname):
    folder = '/home/marysia/thesis/results/nodule_img/'
    fname = folder + fname + '.png'
    data = array[6, :, :]

    start_unit = -1000
    end_unit = 300
    data = 2 * (data.astype(np.float32) - start_unit) / (end_unit - start_unit) - 1
    img = data
    img[img < -1] = -1
    img[img > 1] = 1
    img = (img + 1) * 127.5
    cv2.imwrite(fname, img)

a = np.load('/home/marysia/data/thesis/patches/lidc-localization-patches/positive_patches.npz')['meta']
b = np.load('/home/marysia/data/thesis/patches/lidc-localization-patches/positive_patches.npz')['data']
texture = [elem['annotation-metadata']['texture'] for elem in a]
avgs = [np.mean(elem) for elem in texture]

ggo = [i for i, x in enumerate(avgs) if x == 2.0]
solid = [i for i, x in enumerate(avgs) if x == 4.0]
# #
#
# for i in xrange(10):
#
#     array_to_png(b[ggo[i]], 'ggo-'+str(i))
#     array_to_png(b[solid[i]], 'solid-'+str(i))


print('Done')
