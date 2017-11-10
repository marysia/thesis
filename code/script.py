import numpy as np
import os
import cv2
import glob
import pickle
from utils.generic import create_submission
from utils.config import RESULTSDIR

def get_png(i, data, scope):
    img = data[i, :, :, :]
    img[img < -1] = -1
    img[img > 1] = 1
    img = (img + 1) * 127.5
    for i in xrange(13):
        im = img[i, :, :]
        fname = '/home/marysia/tmp/' + scope + '_' + str(i) + '.png'
        cv2.imwrite(fname, im)


def create_frocs():
    training_set_sizes = [30, 300, 3000, 30000]
    groups = ['Z3', 'C4h', 'D4h', 'O']
    groups = ['O']
    for size in training_set_sizes:
        for group in groups:
            cmd = 'python noduleCADEvaluationLUNA16.py --constraints %d %s' % (size, group)
            os.system(cmd)

def submission_from_pickle(fname):
    with open(fname, 'r') as f:
        pi = pickle.load(f)

    symmetry = True if 'test-symmetry-predictions' in pi.keys() else False
    create_submission(pi, symmetry)

def create_submissions():

    #runs = ['wwddyi', 'vyjbtr', 'pkoejc', 'ofqvfn', 'lxfhyq', 'kbekjr', 'csbsan']
    runs = ['mlkbrc', 'ddgmqr']
    for r in runs:
        fnames = glob.glob(os.path.join(RESULTSDIR, 'pickles', r + '*pkl'))
        print fnames

        for fname in fnames:
            try:
                submission_from_pickle(fname)
            except:
                print fname

def create_best_submissions():
    fnames = glob.glob(os.path.join(RESULTSDIR, 'pickles', '*'))
    for fn in fnames:
        print fn
        with open(fn, 'r') as f:
            pi = pickle.load(f)
        create_submission(pi, True)
        create_submission(pi, False)


create_frocs()