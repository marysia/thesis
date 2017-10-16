
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


def loss_plot(identifier, mode, losstype):
    print losstype
    fname = glob.glob(os.path.join(RESULTSDIR, 'pickles', identifier + '*'))[0]

    with open(fname, 'rb') as f:
        meta = pickle.load(f)
    fname = '-'.join([meta['log-identifier'], meta['name'], meta['group'], str(meta['training-set-samples'])])

    # epochs
    if mode == 'epochs':
        y_train = meta['train-loss']
        y_val = meta[losstype]
        step_size = len(meta['train-loss']) / 10
        x_train = list(xrange(1, len(meta['train-loss'])+1))
        x_val = list(xrange(1, len(meta[losstype])+1))
    elif mode == 'batches':
        cost_batch = 10
        batch_size = meta['batch-size']
        nr_epochs = (len(meta['train-loss']) * cost_batch * batch_size) / meta['training-set-samples']
        save_step = len(meta['train-loss']) / nr_epochs
        y_train = []
        y_val = []

        for i in xrange(0, len(meta['train-loss']), save_step):
            y_train.append(np.mean(meta['train-loss'][i:(i+save_step)]))
            y_val.append(np.mean(meta[losstype][i:(i+save_step)]))
        print y_train[0:3]
        x_train = list(xrange(1, nr_epochs+1))
        x_val = list(xrange(1, nr_epochs+1))
        step_size = 1

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
    mode = sys.argv[2]
    losstype = 'val-loss' if len(sys.argv) > 3 and sys.argv[3] == 'val' else 'test-loss'
    print(identifier)
    loss_plot(identifier, mode, losstype)