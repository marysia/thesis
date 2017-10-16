''' Model-related helper functions '''

import sys

import tensorflow as tf
import matplotlib as mpl

mpl.use('Agg')
import pylab as plt

sys.path.append('../../')
from utils.config import RESULTSDIR

def error_plot(errors, epoch_interval):
    train_errors = [elem[0] for elem in errors]
    val_errors = [elem[1] for elem in errors]

    x = list(xrange(0, epoch_interval * len(errors), epoch_interval))

    plt.plot(x, train_errors, label='Train error')
    plt.plot(x, val_errors, label='Val errors')
    plt.legend(loc='best')
    plt.title('Train/val errors')
    plt.savefig(RESULTSDIR + '/errors.png')

def total_parameters():
    '''
    Returns the total number of trainable parameters.
    '''
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def progress(prefix, step, steps):
    ''' Progress bar implementation.
    Determines output string that simulates a progress bar
    based on current epoch and step, and total number of epochs and steps.
    Flushes this string to stdout, overwriting previous output.
    E.g. Epoch 1 of 15: [=====     ] 50%
    '''
    completed = int((step / float(steps)) * 100)
    progress_str = '[%s%s] %d%%' % ('=' * completed, ' ' * (100 - completed), completed)
    sys.stdout.write(prefix + progress_str)
    sys.stdout.flush()


def pretty_print_confusion_matrix(confusion_matrix):
    str = 'Confusion matrix: \n'
    str += '\t TN: %d \t FP: %d \n' % (confusion_matrix[0][0], confusion_matrix[0][1])
    str += '\t FN: %d \t TP: %d ' % (confusion_matrix[1][0], confusion_matrix[1][1])
    return str
