import argparse

from preprocessing.types import DataTypes
from models.model_3d import CNN
from models.util.helpers import total_parameters
from models.util.metrics import get_accuracy, get_confusion_matrix
from utils.control import ProgramEnder
from utils.logger import Logger
from preprocessing.generic import data_metrics
from utils.generic import save_meta
from utils.config import LOGDIR
# global program ender
ender = ProgramEnder()
LOGDIR = '/home/marysia/thesis/logs/'


def main(args):
    data = DataTypes(args)
    log = Logger(args, LOGDIR, logfolder='composition', logname=args.log)
    data_metrics(data, log)

    tot_params = 0

    for group in args.groups:
        graph = CNN(name='CNN', group=group.upper(), data=data)

        # calculate the number of parameters
        model_parameters = total_parameters() - tot_params
        tot_params += model_parameters
        log.info('Number of parameters is... %d ' % (model_parameters))

        graph.train(args, ender=ender, log=log)

        graph.set_meta(0, model_parameters)

        save_meta(graph.meta)

        log.result('Validation set accuracy: %.3f ' % get_accuracy(graph.meta, 'val', args.symmetry))
        log.result('Test set accuracy: %.3f' % get_accuracy(graph.meta, 'test', args.symmetry))

        conf, acc = get_confusion_matrix(graph.meta, 'test', args.symmetry)
        log.result(conf)
        log.result(acc)


    log.finalize(ender.terminate, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--discard", action="store_true")
    parser.add_argument("--log", nargs="?", type=str, default="composition")

    # model
    parser.add_argument("--groups", nargs="+", default=['Z3'])
    parser.add_argument("--repeats", nargs="?", type=int, default=1)

    parser.add_argument("--mode", nargs="?", type=str, default="epochs", choices=["epochs", "time", "converge"])
    parser.add_argument("--mode_param", nargs="?", type=int, default=5)
    parser.add_argument("--augment", nargs="+", default=[])

    # data arguments
    parser.add_argument("--train", nargs="?", type=str, default='nlst-balanced')
    parser.add_argument("--val", nargs="?", type=str, default='nlst')
    parser.add_argument("--test", nargs="?", type=str, default="lidc")
    #parser.add_argument("--samples", nargs="+", default=[5000])
    parser.add_argument("--shape", nargs="+", default=[12, 72, 72])

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--symmetry", action="store_true")
    args = parser.parse_args()
    print(args)

    main(args)