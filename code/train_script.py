import utils.environment
import os
import sys
import argparse

from preprocessing.patches import DataPatches
from preprocessing.mnist import DataMNIST
from preprocessing.generic import data_metrics
from models.model_2d import ConvolutionalModel1, GConvModel1, Resnet, GResnet, Z2CNN, P4CNN, P4CNNDropout
from models.model_3d import Z3CNN, GCNN, Resnet, GResnet
from utils.control import ProgramEnder
from utils.logger import Logger
from models.util.helpers import total_parameters

models2d = {
#    'Resnet1': Resnet#,
#    'GResnet1': GResnet
#    'Conv1': ConvolutionalModel1,
#    'Gconv1': GConvModel1,
     'Z2CNN': Z2CNN,
     'P4CNN': P4CNN,
     'P4CNNDropout': P4CNNDropout
}

models3d = {
#     'Z3CNN': Z3CNN,
     'GCNN': GCNN
#    'Resnet': Resnet,
#    'GResnet': GResnet
}

# global program ender
ender = ProgramEnder()

def train(data, log, models, args):
    data_metrics(data, log)
    tot_params = 0

    # train and evaluate each model
    for name, model in models.items():
        if not ender.terminate:
            log.info('\n\nStarting model ' + name)
            graph = model(model_name=name, data=data, ender=ender, log=log, verbose=args.verbose)
            graph.build_model()

            # calculate parameters
            model_params = total_parameters() - tot_params
            tot_params += model_params
            log.info('Model: Number of parameters is... ' + str(model_params))

            # train
            #graph.train(mode=args.mode, mode_param=args.mode_param, save_step=args.save_step)
        else:
            log.info('Program was terminated. Model %s will not be trained and/or evaluated.' % name)
    log.info('Executed all steps in script.\n')

def mnist(args):
    log = Logger('/home/marysia/thesis/logs/', logname='mnist_'+args.log)
    log.info('Executing mnist script with the following models: %s' % str(models2d.keys()))
    data = DataMNIST()
    train(data, log, models2d, args)

def patches_2d(args):
    log = Logger('/home/marysia/thesis/logs/', logname='patches2d_'+args.log)
    log.info('Executing patches 2d script with the following models: %s' % str(models2d.keys()))
    data = DataPatches(small=args.smalldata, shape=(1, 30, 30))
    train(data, log, models2d, args)

def patches_3d(args):
    log = Logger('/home/marysia/thesis/logs/', logname='patches3d_'+args.log)
    log.info('Executing patches 3d script with the following models: %s' % str(models3d.keys()))
    data = DataPatches(small=args.smalldata, shape=(8, 30, 30))
    train(data, log, models3d, args)

if __name__ == "__main__":
    argument_function_mapping = dict(zip(["mnist", "patches2d", "patches3d"], [mnist, patches_2d, patches_3d]))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="?", type=str, default="patches2d", choices=argument_function_mapping.keys())
    parser.add_argument("--log", nargs="?", type=str, default="log")
    parser.add_argument("--mode", nargs="?", type=str, default="epochs", choices=["epochs", "time", "converge"])
    parser.add_argument("--mode_param", nargs="?", type=int,  default=5)
    parser.add_argument("--save_step", nargs="?", type=int, default=1)
    parser.add_argument("--smalldata", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    print(args)
    print(argument_function_mapping)

    argument_function_mapping[args.data](args)