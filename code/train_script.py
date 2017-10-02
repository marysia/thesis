import argparse

import numpy as np

from models.model_3d import CNN, WideBoostingNetwork
from models.util.helpers import total_parameters, create_submission, pretty_print_confusion_matrix
from models.util.metrics import ensemble_submission
from preprocessing.generic import data_metrics
from preprocessing.patches import DataPatches
from utils.control import ProgramEnder
from utils.generic import log_time, submission, get_best_predictions
from utils.logger import Logger
from noduleCADEvaluationLUNA16 import create_froc_curve
models3d = {
#     'WBN': WideBoostingNetwork
     'CNN': CNN
}

# global program ender
ender = ProgramEnder()
logdir = '/home/marysia/thesis/logs/'


def get_average(evaluation, target):
    values = [elem[target] for elem in evaluation]
    return np.mean(values), np.std(values)


def average_evaluation(evaluation, log):
    if len(evaluation) > 1:
        for key in evaluation[0].keys():
            if key != 'results' and key != 'weight':
                mean, std = get_average(evaluation, key)
                log.result('- %s \t %.2f (~%.2f)' % (key, mean, std))


def ensemble(results, model_weights, data, log, submission):
    probabilities, conf_matrix, a, p, s, fp_rate = ensemble_submission(results, model_weights, data.test)

    log.info('\n Ensemble model: ')
    log.result('Acc: %.2f, Prec: %.2f, Sensitivity: %.2f, FP-rate: %.2f' % (a, p, s, fp_rate))
    log.result(pretty_print_confusion_matrix(conf_matrix))

    if submission:
        create_submission('ensemble', log, data.test, probabilities)


def train(data, log, trainmodels, args, augmentation):
    log_time(log, args, trainmodels)  # current time + expected end time
    data_metrics(data, log)  # metrics of current dataset

    # initialize
    tot_params = 0

    # Go through all models, i.e. CNN, Resnet, MultiDim...
    for name, model in trainmodels.items():

        # through all groups, i.e. Z3, O, B
        for group in args.groups:

            model_evaluations = []

            # train all model/group combinations args.repeats times.
            for _ in xrange(args.repeats):

                # only train if no termination signal has been sent.
                if not ender.terminate:

                    # initialize
                    log.info('\n\nStarting model %s with group %s.' % (name, group))
                    graph = model(name=name, group=group.upper(), data=data)

                    # calculate parameters
                    model_params = total_parameters() - tot_params
                    tot_params += model_params
                    log.info('Model: Number of parameters is... ' + str(model_params))

                    # train
                    graph.train(args, augmentation=augmentation, ender=ender, log=log)
                    # save some results.
                    model_evaluations.append(graph.evaluation)

                else:
                    log.info('Program was terminated. Model %s will not be trained and/or evaluated.' % name)
            log.info('Evaluation of %d repeats of %s with group %s, train set size: %d.' % (args.repeats, name,
                                                                                            group, data.train.samples))
            average_evaluation(model_evaluations, log)

            if args.submission:
                result_files = []
                prefix = args.log + '-' if args.log != '' else ''
                for i, evaluation in enumerate(model_evaluations):
                    modelname = '%s%s-%s-%d-%d' % (prefix, name, group, data.train.samples, i+1)
                    print(modelname)
                    fname = submission(args.test, data.test, evaluation['results'], modelname, log)
                    result_files.append(fname)

                if args.ensemble:
                    predictions = get_best_predictions(model_evaluations, True)
                    fname = submission(args.test, data.test, predictions, '%s%s-ensemble-%d' % (prefix, name, args.repeats),
                                           log)
                    result_files.append(fname)
                #create_froc_curve(result_files)

    log.info('Executed all steps in script.\n')


def main(args):
    """
    Main function. Creates the logger, augmentation steps and data.

    Loops through all sample sizes provides in reverse order, resizing the train data
    to a smaller set each time, and training.
    """
    broken = False
    log = Logger(args, logdir, logfolder='patches3d', logname=args.log)
    data = DataPatches(args)

    # go through sample size in reverse order: [10.000, 1.000, 100, 10]
   # try:
    samples = args.samples
    samples.sort(reverse=True)
    for sample_size in samples:
        data.train.resize(int(sample_size))  # resize the training data to sample_size
 #       train(data, log, models3d, args, args.augment)
    #except:
    #    broken = True

    log.finalize(ender.terminate, broken)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--log", nargs="?", type=str, default="log")

    # model
    parser.add_argument("--groups", nargs="+", default=['Z3'])
    parser.add_argument("--repeats", nargs="?", type=int, default=1)

    parser.add_argument("--mode", nargs="?", type=str, default="epochs", choices=["epochs", "time", "converge"])
    parser.add_argument("--mode_param", nargs="?", type=int, default=5)
    parser.add_argument("--save_step", nargs="?", type=int, default=100)
    parser.add_argument("--augment", nargs="+", default=[])

    # data arguments
    parser.add_argument("--train", nargs="?", type=str, default='nlst-balanced')
    parser.add_argument("--val", nargs="?", type=str, default='nlst-unbalanced')
    parser.add_argument("--test", nargs="?", type=str, default="lidc-localization")
    parser.add_argument("--samples", nargs="+", default=[5000])
    parser.add_argument("--shape", nargs="+", default=[12, 72, 72])
    parser.add_argument("--zoomed", action="store_true")

    # flags

    parser.add_argument("--reinforce", action="store_true")
    parser.add_argument("--save_model", action="store_true")

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--symmetry", action="store_true")

    args = parser.parse_args()
    print(args)

    main(args)
