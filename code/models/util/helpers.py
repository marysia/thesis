''' Model-related helper functions '''
import numpy as np
import tensorflow as tf
import sys
import random

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
    progress_str = '[%s%s] %d%%' % ('='*completed, ' '*(100-completed), completed)
    sys.stdout.write(prefix + progress_str)
    sys.stdout.flush()