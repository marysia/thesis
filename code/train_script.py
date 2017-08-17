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

    # train and evaluate each model
    for name, model in models.items():
        if not ender.terminate:
            log.info('\n\nStarting model %s' % name)
            graph = model(model_name=name, data=data, ender=ender, log=log, augmentation=augmentation,
                          submission=args.submission, verbose=args.verbose)
            graph.build_model()

            # calculate parameters
            model_params = total_parameters() - tot_params
            tot_params += model_params
            log.info('Model: Number of parameters is... ' + str(model_params))

            # train
            graph.train(args)
            #graph.train(mode=args.mode, mode_param=args.mode_param, save_step=args.save_step, reinforce=args.reinforce,
            #            save_model=args.save_model)
        else:
            log.info('Program was terminated. Model %s will not be trained and/or evaluated.' % name)
    log.info('Executed all steps in script.\n')


def mnist(args):
    ''' MNIST train '''
    augmentation = "rotation2d" if args.augment else ""
    log = Logger(args, logdir, logfolder='mnist', logname=args.log)
    log.info('Executing mnist script with the following models: %s' % str(models2d.keys()))
    data = DataMNIST(shape=(7, 7))
    train(data, log, models2d, args, augmentation)
    log.finalize(ender.terminate)


def patches_2d(args):
    ''' 2D variant of patches '''
    augmentation = "rotation2d" if args.augment else ""
    log = Logger(args, logdir, logfolder='patches2d', logname=args.log)
    log.info('Executing patches 2d script with the following models: %s' % str(models2d.keys()))
    data = DataPatches(shape=(1, 15, 15))
    train(data, log, models2d, args, augmentation)
    log.finalize(ender.terminate)


def patches_3d(args):
    ''' 3D variant of patches '''
    augmentation = "rotation3d" if args.augment else ""
    log = Logger(args, logdir, logfolder='patches3d', logname=args.log)
    log.info('Executing patches 3d script with the following models: %s' % str(models3d.keys()))
    data = DataPatches(shape=(8, 30, 30))
    train(data, log, models3d, args, augmentation)
    log.finalize(ender.terminate)


if __name__ == "__main__":
    argument_function_mapping = dict(zip(["mnist", "patches2d", "patches3d"], [mnist, patches_2d, patches_3d]))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="?", type=str, default="patches2d", choices=argument_function_mapping.keys())
    parser.add_argument("--log", nargs="?", type=str, default="log")
    parser.add_argument("--mode", nargs="?", type=str, default="epochs", choices=["epochs", "time", "converge"])
    parser.add_argument("--mode_param", nargs="?", type=int, default=5)
    parser.add_argument("--save_step", nargs="?", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--reinforce", action="store_true")
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()
    print(args)

    argument_function_mapping[args.data](args)
