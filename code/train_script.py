import argparse

from models.model_2d import Z2CNN
from models.model_3d import Z3CNN, GCNN
from models.util.helpers import total_parameters, create_submission, pretty_print_confusion_matrix
from models.util.metrics import ensemble_submission
from preprocessing.generic import data_metrics
from preprocessing.mnist import DataMNIST
from preprocessing.patches import DataPatches
from utils.control import ProgramEnder
from utils.generic import log_time
from utils.logger import Logger

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
    'GCNN': GCNN,
    #    'Z3MultiDim': Z3MultiDim,
    #    'GMultiDim': GMultiDim
    #    'Z3Resnet': Z3Resnet,
    #    'GResnet': GResnet
}

# global program ender
ender = ProgramEnder()
logdir = '/home/marysia/thesis/logs/'


def ensemble(results, model_weights, data, log, submission):
    probabilities, conf_matrix, a, p, s, fp_rate = ensemble_submission(results, model_weights, data.test)

    log.info('\n Ensemble model: ')
    log.result('Acc: %.2f, Prec: %.2f, Sensitivity: %.2f, FP-rate: %.2f' % (a, p, s, fp_rate))
    log.result(pretty_print_confusion_matrix(conf_matrix))

    if submission:
        create_submission('ensemble', log, data.test, probabilities)


def train(data, log, trainmodels, args, augmentation):
    log_time(log, args, trainmodels)
    data_metrics(data, log)
    tot_params = 0
    results = []
    model_weights = []

    # train and evaluate each model
    for name, model in trainmodels.items():
        if not ender.terminate:
            for _ in xrange(args.repeats):
                log.info('\n\nStarting model %s' % name)
                graph = model(name=name, data=data)

                # calculate parameters
                model_params = total_parameters() - tot_params
                tot_params += model_params
                log.info('Model: Number of parameters is... ' + str(model_params))

                # train
                graph.train(args, augmentation=augmentation, ender=ender, log=log)
                results.append(graph.results)
                model_weights.append(graph.model_weight)
        else:
            log.info('Program was terminated. Model %s will not be trained and/or evaluated.' % name)

    if args.ensemble:
        ensemble(results, model_weights, data, log, args.submission)

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
    augmentation = ['scale', 'flip', 'rotate'] if args.augment else []
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
    parser.add_argument("--repeats", nargs="?", type=int, default=1)

    # data arguments
    parser.add_argument("--data", nargs="?", type=str, default="patches3d", choices=argument_function_mapping.keys())
    parser.add_argument("--train", nargs="?", type=str, default='nlst-balanced')
    parser.add_argument("--val", nargs="?", type=str, default='nlst-unbalanced')
    parser.add_argument("--test", nargs="?", type=str, default="lidc-localization")
    parser.add_argument("--shape", nargs="?", type=tuple, default=(8, 30, 30))
    parser.add_argument("--samples", nargs="?", type=int, default=-100)
    parser.add_argument("--zoomed", action="store_true")

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--reinforce", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--ensemble", action="store_true")

    args = parser.parse_args()
    print(args)

    augmentation, log, data, models = argument_function_mapping[args.data](args)

    log.info('Executing %s script with the following models: %s' % (args.data, models.keys()))
    train(data, log, models, args, augmentation)
    log.finalize(ender.terminate)
