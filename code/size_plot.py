
import matplotlib as mpl

mpl.use('Agg')
import pylab as plt
import pickle
import glob
import sys
import os
import numpy as np

from utils.config import RESULTSDIR

def create_plot():


    x_axis = [1000, 5000, 10000, 15000]
    y_axis = [.65, .70, .75, .80, .85]
    y_axis_z3 = [.723, .766, .781, .792]
    y_axis_b = [.769, .825, .828, .823]

    plt.plot(x_axis, y_axis_z3, label='Z3')
    plt.plot(x_axis, y_axis_b, label='B')
    plt.xticks(x_axis)
    plt.yticks(y_axis)

    plt.legend(loc='best')
    plt.title('tryout')
    plt.savefig('val.png')


def loss_plot(meta):

    fname = '-'.join([meta['log-identifier'], meta['name'], meta['group'], str(meta['training-set-samples']), str(meta['log-identifier'])])

    # epochs
    y_train = meta['train-loss']
    y_val = meta['val-loss']
    step_size = len(meta['train-loss']) / 10
    x_train = list(xrange(1, len(meta['train-loss'])+1))
    x_val = list(xrange(1, len(meta['val-loss'])+1))

    # print x_train, y_train
    # print x_val, y_val
    plt.plot(x_train, y_train, label='train-loss')
    plt.plot(x_val, y_val, label='val-loss')

    print(len(x_train))
    print(len(x_val))
    plt.xticks(list(xrange(0, len(x_train)+1, step_size)))



    title = 'Train and validation set loss for %s-%s on %d samples' % (meta['group'], meta['name'], meta['training-set-samples'])

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig(os.path.join(RESULTSDIR, 'plots', '%s.png' % fname))

if __name__ == '__main__':
    identifier = sys.argv[1]

    fnames = glob.glob(os.path.join(RESULTSDIR, 'pickles', identifier + '*'))

    for fname in fnames:
        print(fname)
        with open(fname, 'rb') as f:
            meta = pickle.load(f)
        loss_plot(meta)