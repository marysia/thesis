import model

import sys
import os
from logger import Logger
from training import Train
from data import load_lidc, load_candidates

# shortcut for running from PyCharm (for debugging purposes)
if len(sys.argv) == 1:
    args = ['script.py', 'lidc', '5']
else:
    args = sys.argv
# initialize logger
log = Logger(args, depth=3)
log.backup_additional(['model.py', 'logger.py', 'training.py', 'data.py'])

# global variables
ordering = 'th'
dataset, epochs = args[-2:]

# get data
if dataset == 'lidc':
    xtrain, ytrain, xtest, ytest = load_lidc(ordering)
if dataset == 'candidates':
    xtrain, ytrain, xtest, ytest = load_candidates(ordering)


def train_model(network):
    ''' Trains and logs result for each model. '''
    log.info('Starting ' + network.name)
    # compile and train
    train = Train(network.model, epochs=int(epochs))
    train.compile_model()
    train.train_model(xtrain, ytrain, xtest, ytest)

    # evaluate and log
    results = train.evaluate(xtest, ytest)
    res, accuracy, sensitivity, fpr = train.get_results(xtest, ytest)

    log.result(results)
    log.result('Accuracy: ' + str(accuracy))
    log.result('Table: ' + str(res))
    log.result('Sensitivity: ' + str(sensitivity))
    log.result('False Positive Rate: ' + str(fpr))

    # save model
    train.model.save(os.path.join(log.path, network.name + '.h5'))
    log.info('Model succesfully saved at ' + os.path.join(log.path, network.name + '.h5'))

    log.info('----------------\n')


# ---- Main script --- #
print('Starting.')
if ordering == 'tf':
    cnn3d = model.CNN_3D()
    zcnn = model.Zuidhof()
if ordering == 'th':
    cnn3d = model.CNN_3D(input_shape=(1, 7, 72, 72))
    zcnn = model.Zuidhof(input_shape=(1, 7, 72, 72))

train_model(cnn3d)
#train_model(zcnn)

log.copy()
print('Done.')


