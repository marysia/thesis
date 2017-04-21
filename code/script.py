import time
import sys
from control import ProgramEnder
from logger import Logger


import model
import sys
import os
import datetime
import numpy as np
from logger import Logger
from training import Train
from data import load_lidc, load_candidates
import argparse
from keras import backend as K

def get_network(name, input_shape):
    if name == 'znet':
        network = model.ZuidhofRN(input_shape=input_shape)
    elif name == 'zcnn':
        network = model.Zuidhof(input_shape=input_shape)
    elif name == 'resnet':
        network = model.Resnet3D(input_shape=input_shape)
    return network

def train_model(network, data, params, log, accuracies, postfix=''):
    ''' Trains and logs result for each model. '''
    log.info('Starting ' + network.name)
    # compile and train
    train = Train(network.model, epochs=params['epochs'], batch_size=params['batch_size'])
    train.compile_model()
    train.train_model(data['xtrain'], data['ytrain'], data['xtest'], data['ytest'])

    # evaluate and log
    res, accuracy, sensitivity, fpr = train.get_results(data['xtest'], data['ytest'])
    accuracies.append(accuracy)

    log.result('Accuracy: ' + str(accuracy))
    log.result('Table: ' + str(res))
    log.result('Sensitivity: ' + str(sensitivity))
    log.result('False Positive Rate: ' + str(fpr))

    # save model
    train.model.save(os.path.join(log.path, network.name + postfix + '.h5'))
    log.info('Model successfully saved at ' + os.path.join(log.path, network.name + '.h5'))

    log.info('----------------\n')
    return accuracies


if __name__ == '__main__':
    print('HI STARTING NOW!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lidc')
    parser.add_argument('--order', type=str, default='tf')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--network', type=str, default='znet')
    args = vars(parser.parse_args())
    log_args = [__file__] + args.values()

    log = Logger(sys.argv, depth=3)
    log.info(args)
    log.backup_additional(['resnet/model.py', 'resnet/data.py', 'resnet/training.py'])
    #ender = ProgramEnder()

    # get data
    if args['dataset'] == 'lidc':
        xtrain, ytrain, xtest, ytest = load_lidc(args['order'])
    if args['dataset'] == 'candidates':
        xtrain, ytrain, xtest, ytest = load_candidates(args['order'])
    # dict for easier passing
    data = {'xtrain': xtrain, 'xtest': xtest, 'ytrain': ytrain, 'ytest': ytest}

    # set input shape
    if args['order'] == 'th':
        input_shape = (1, 7, 72, 72)
    else:
        input_shape = (7, 72, 72, 1)


    # -- main script -- #
    start = datetime.datetime.now()
    accuracies = []

    i = 0
    # while not ender.terminate and i < args['repeats']:
    #     log.info('Attempt ' + str(i))
    #
    #     network = get_network(args['network'].lower(), input_shape)
    #     train_model(network, data, args, log, accuracies, postfix='_' + str(i))
    #     i += 1
    #
    # if ender.terminate:
    #     log.error('Program was terminated. Exiting gracefully.')


    while i < args['repeats']:
        network = get_network(args['network'].lower(), input_shape)
        train_model(network, data, args, log, accuracies, postfix='_'+str(i))
        i += 1

    if args['repeats'] > 1:
        log.result('Mean of accuracies: ' + str(np.mean(accuracies)))
        log.result('STD of accuracies: ' + str(np.std(accuracies)))
        log.result('Max of accuracies: ' + str(np.max(accuracies)))
        log.result('Min of accuracies: ' + str(np.min(accuracies)))

    end = datetime.datetime.now()
    log.info('Running script took ' + str((end-start).seconds) + ' seconds.\n')
    log.copy()





