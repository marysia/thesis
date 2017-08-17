''' Model-related helper functions '''
import numpy as np
import tensorflow as tf
import sys
import os
import csv

results_folder = '/home/marysia/thesis/results'

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

def pretty_print_confusion_matrix(confusion_matrix):
    str = 'Confusion matrix: \n'
    str += '\t TN: %d \t FP: %d \n' % (confusion_matrix[0][0], confusion_matrix[0][1])
    str += '\t FN: %d \t TP: %d ' % (confusion_matrix[1][0], confusion_matrix[1][1])
    return str


def create_submission(modelname, log, data, results):
    # voor elk result:
        # get identifier, bepaalde class, probability, actual class
    submission_fname = os.path.join(results_folder, 'submissions', '%s-%s-submission.csv' % (log.runid, modelname))
    li = [['identifier', 'predicted_class', 'probability', 'actual_class']]

    differences = [abs(r[0]-r[1]) for r in results]
    max_diff, min_diff = max(differences), min(differences)
    for i, result in enumerate(results):
        id = data.id[i]
        pred_class = np.argmax(result)
        prob_class = (differences[i] - min_diff) / (max_diff - min_diff)
        act_class = np.argmax(data.y[i])

        li.append([id, pred_class, prob_class, act_class])

    probs = [elem[2] for elem in li]
    probs = probs[1:]
    log.info('Probabilities. range: %.2f-%.2f, mean: %.2f (~%.2f)' % (np.min(probs), np.max(probs),
                                                                       np.mean(probs), np.std(probs)))

    with open(submission_fname, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(li)
