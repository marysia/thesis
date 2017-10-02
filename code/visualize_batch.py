import argparse
from preprocessing.patches import DataPatches
from models.util.augmentation import augment_batch
import time
import cv2
import numpy as np

def augment(x, transformations):
    a = time.time()
    augmented_batch = augment_batch(x, transformations, [12, 72, 72], keep_prob=0.0)
    print(transformations, time.time() - a)
    return augmented_batch

def make_grid(x):
    a = x[0, 6, :, :, 0]
    b = x[1, 6, :, :, 0]
    c = x[2, 6, :, :, 0]
    new_x = np.concatenate([a, b, c])

    a = x[3, 6, :, :, 0]
    b = x[4, 6, :, :, 0]
    c = x[5, 6, :, :, 0]
    tmp = np.concatenate([a, b, c])
    new_x = np.concatenate([new_x.transpose(), tmp.transpose()])

    a = x[6, 6, :, :, 0]
    b = x[7, 6, :, :, 0]
    c = x[8, 6, :, :, 0]

    tmp = np.concatenate([a, b, c])
    new_x = np.concatenate([new_x, tmp.transpose()])
    new_x = new_x.transpose()
    return new_x


def visualize(x, fname):
    # x = x[:, 6, :, :, 0]
    # x = x.reshape((360, 360))
    #x = x[0, 6, :, :, 0]
    x = make_grid(x)
    x[x < -1] = -1
    x[x > 1] = 1
    x = (x + 1) * 127.5
    cv2.imwrite(fname, x)

def main(args):
    data = DataPatches(args)
    x, y = data.train.get_next_batch(0, 9)
    orig_x = augment(x, [])
    # transformations = ['scale', 'flip', 'rotate', 'noise']
    aug_x = augment(x, ['blur'])
    #aug_x = augment(x, ['flip'])
    #aug_x = augment(x, ['rotate'])
    #aug_x = augment(x, ['noise'])
    # augment(x, transformations)
    visualize(orig_x, 'nodule_original.png')
    visualize(aug_x, 'nodule_augmented.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", nargs="?", type=str, default='nlst-balanced')
    parser.add_argument("--val", nargs="?", type=str, default='empty')
    parser.add_argument("--test", nargs="?", type=str, default="empty")
    parser.add_argument("--samples", nargs="+", default=[5000])
    parser.add_argument("--shape", nargs="+", default=[8, 30, 30])
    parser.add_argument("--zoomed", action="store_true")


    args = parser.parse_args()
    print(args)

    main(args)
    print('Done')
