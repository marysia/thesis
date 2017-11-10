import argparse

from models.model_3d import WideBoostingNetwork, CNN
from models.util.helpers import total_parameters
from preprocessing.generic import data_metrics
from preprocessing.patches import DataPatches
from utils.control import ProgramEnder
from utils.generic import log_time, save_meta, create_submission
from utils.logger import Logger
from utils.config import LOGDIR

models3d = {
    #'WBN': WideBoostingNetwork#,
    'CNN': CNN
}

# global program ender
ender = ProgramEnder()


def train(data, log, trainmodels, args):
    log_time(log, args, trainmodels)
    data_metrics(data, log)

    tot_params = 0

    for name, model in trainmodels.items():
        for group in args.groups:
            for iteration in xrange(args.iteration, (args.repeats+args.iteration)):
                if not ender.terminate:
                    # initialize the model
                    log.info('\n\nStarting model %s with group %s, iteration %d.' % (name, group, iteration), time=True)
                    graph = model(name=name, group=group.upper(), data=data)

                    # calculate the number of parameters
                    model_parameters = total_parameters() - tot_params
                    tot_params += model_parameters
                    log.info('Number of parameters is... %d ' % (model_parameters))

                    #train
                    graph.train(args, ender=ender, log=log)

                    # save results
                    graph.set_meta(iteration, model_parameters)
                    save_meta(graph.meta)

                    if args.submission:
                        create_submission(graph.meta, args.symmetry)

                else:
                    log.info('Program was terminated. Model %s will not be trained and/or evaluated.' % name, time=True)

    log.info('Executed all steps in script.\n')


def main(args):
    """
    Main function. Creates the logger, augmentation steps and data.

    Loops through all sample sizes provides in reverse order, resizing the train data
    to a smaller set each time, and training.
    """
    broken = False
    log = Logger(args, LOGDIR, logfolder='patches3d', logname=args.log)
    data = DataPatches(args)

    # go through sample size in reverse order: [10.000, 1.000, 100, 10]
    samples = args.samples
    samples.sort(reverse=True)
    for sample_size in samples:
        data.train.resize(int(sample_size))  # resize the training data to sample_size
        train(data, log, models3d, args)

    log.finalize(ender.terminate, broken)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--discard", action="store_true")
    parser.add_argument("--log", nargs="?", type=str, default="log")

    # model
    parser.add_argument("--groups", nargs="+", default=['Z3'])
    parser.add_argument("--iteration", nargs="?", type=int, default=0)
    parser.add_argument("--repeats", nargs="?", type=int, default=1)

    parser.add_argument("--mode", nargs="?", type=str, default="epochs", choices=["epochs", "time", "batches"])
    parser.add_argument("--mode_param", nargs="?", type=int, default=5)
    parser.add_argument("--augment", nargs="+", default=[])

    # data arguments
    parser.add_argument("--train", nargs="?", type=str, default='nlst-balanced')
    parser.add_argument("--val", nargs="?", type=str, default='nlst-unbalanced')
    parser.add_argument("--test", nargs="?", type=str, default="lidc-localization")
    parser.add_argument("--samples", nargs="+", default=[5000])
    parser.add_argument("--shape", nargs="+", default=[12, 72, 72])
    parser.add_argument("--save_fraction", nargs="?", type=float, default=.7)

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--symmetry", action="store_true")

    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=0.0001)

    # old
    parser.add_argument("--zoomed", action="store_true")

    args = parser.parse_args()
    print(args)

    main(args)
