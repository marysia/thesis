import utils.environment
import time
import datetime
import argparse

from preprocessing.patches import DataPatches
from preprocessing.mnist import DataMNIST
from preprocessing.generic import data_metrics
from models.model_2d import ConvolutionalModel1, GConvModel1, Resnet, GResnet, Z2CNN, P4CNN, P4CNNDropout
from models.model_3d import Z3CNN, GCNN, Z3Resnet, GResnet, Z3MultiDim, GMultiDim
from utils.control import ProgramEnder
from utils.logger import Logger
from models.util.helpers import total_parameters

import numpy as np
models2d = {
    #    'Resnet1': Resnet#,
    #    'GResnet1': GResnet
    #    'Conv1': ConvolutionalModel1,
    #    'Gconv1': GConvModel1,
    'Z2CNN': Z2CNN,
#    'P4CNN': P4CNN,
    #     'P4CNNDropout': P4CNNDropout
}

models3d = {
    'Z3CNN': Z3CNN,
#    'GCNN': GCNN,
#    'Z3MultiDim': Z3MultiDim,
#    'GMultiDim': GMultiDim
#    'Z3Resnet': Z3Resnet,
#    'GResnet': GResnet
}

# global program ender
ender = ProgramEnder()
logdir = '/home/marysia/thesis/logs/'


def get_probabilities(result):
    result = np.array(result)
    differences = [abs(r[0]-r[1]) for r in result]
    max_diff, min_diff = max(differences), min(differences)
    probabilities = [0] * len(differences)
    for i, r in enumerate(result):
        prob = (differences[i] - min_diff) / (max_diff - min_diff)
        if np.argmax(r) == 1:
            probabilities[i] = prob
        else:
            probabilities[i] = 1 - prob
    return probabilities

def evaluate_results(results, data):
    probs = []
    for r in results:
        probs.append(get_probabilities(r))

    probs = np.array(probs)
    avg_probs = np.mean(probs, axis=0)

    predictions = [0 if elem < .5 else 1 for elem in avg_probs ]
    actual = [np.argmax(elem) for elem in data.y]
    correct = 0
    for i in xrange(len(predictions)):
        if predictions[i] == actual[i]:
            correct += 1
    print(correct/float(len(predictions)))

def log_time(log, args, models):
    start_time = time.time() + (2 * 3600)  # system time is two hours behind; adjust.
    start_str = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    log.info('Starting training at: \t %s' % start_str)

    if args.mode == 'time':
        end_time = start_time + ((args.mode_param * 60) * len(models.keys()))
        end_str = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        log.info('Estimated end time: \t %s' % end_str)


def train(data, log, models, args, augmentation):
    log_time(log, args, models)
    data_metrics(data, log)
    tot_params = 0
    repeats = 3
    results = []

    # train and evaluate each model
    for name, model in models.items():
        if not ender.terminate:
            for _ in xrange(repeats):
                log.info('\n\nStarting model %s' % name)
                graph = model(name=name, data=data)
                #graph = model(model_name=name, data=data, ender=ender, log=log, augmentation=augmentation, submission=args.submission, verbose=args.verbose)
                #graph.build_model()

                # calculate parameters
                model_params = total_parameters() - tot_params
                tot_params += model_params
                log.info('Model: Number of parameters is... ' + str(model_params))

                # train
                graph.train(args, augmentation=augmentation, ender=ender, log=log)
                results.append(graph.results)
            evaluate_results(results, data.test)
        else:
            log.info('Program was terminated. Model %s will not be trained and/or evaluated.' % name)
    log.info('Executed all steps in script.\n')


def mnist(args):
    ''' 3D variant of patches '''
    augmentation = "rotation2d" if args.augment else ""
    log = Logger(args, logdir, logfolder='mnist', logname=args.log)
    data = DataMNIST(shape=(7, 7))
    return augmentation, log, data, models3d


def patches_2d(args):
    ''' 2D variant of patches '''
    augmentation = "rotation2d" if args.augment else ""
    log = Logger(args, logdir, logfolder='patches2d', logname=args.log)
    data = DataPatches(args)
    return augmentation, log, data, models2d

def patches_3d(args):
    ''' 3D variant of patches '''
    augmentation = "rotation3d" if args.augment else ""
    log = Logger(args, logdir, logfolder='patches3d', logname=args.log)
    data = DataPatches(args)
    return augmentation, log, data, models3d

if __name__ == "__main__":
    argument_function_mapping = dict(zip(["mnist", "patches2d", "patches3d"], [mnist, patches_2d, patches_3d]))

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", nargs="?", type=str, default="log")
    parser.add_argument("--mode", nargs="?", type=str, default="epochs", choices=["epochs", "time", "converge"])
    parser.add_argument("--mode_param", nargs="?", type=int, default=5)
    parser.add_argument("--save_step", nargs="?", type=int, default=1)

    # data arguments
    parser.add_argument("--data", nargs="?", type=str, default="patches2d", choices=argument_function_mapping.keys())
    parser.add_argument("--train", nargs="?", type=str, default='nlst-balanced')
    parser.add_argument("--val", nargs="?", type=str, default='nlst-unbalanced')
    parser.add_argument("--test", nargs="?", type=str, default="lidc-localization")
    parser.add_argument("--shape", nargs="?", type=tuple, default=(8, 30, 30))

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--reinforce", action="store_true")
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()
    print(args)

    augmentation, log, data, models = argument_function_mapping[args.data](args)

    log.info('Executing %s script with the following models: %s' % (args.data, models.keys()))
    train(data, log, models, args, augmentation)
    log.finalize(ender.terminate)