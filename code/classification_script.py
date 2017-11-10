import argparse
import numpy as np
from preprocessing.composition import DataComposition
from models.model_3d import CNN, WideBoostingNetwork
from models.util.helpers import total_parameters
from models.util.metrics import get_accuracy, get_confusion_matrix
from utils.control import ProgramEnder
from utils.logger import Logger
from preprocessing.generic import data_metrics
from utils.generic import save_meta
from utils.config import LOGDIR

# global program ender
ender = ProgramEnder()

def main(args):
    data = DataComposition(args)
    log = Logger(args, LOGDIR, logfolder='composition', logname=args.log)

    # initialize end results
    final_accuracies = []
    final_confs = np.zeros(shape=(3,3)) if args.foldtype == '3-class' else np.zeros(shape=(2,2))

    for fold in xrange(args.folds):
        # initialize
        fold_accuracies = []
        fold_confs = np.zeros(shape=(3,3)) if args.foldtype == '3-class' else np.zeros(shape=(2,2))

        data.set_fold(fold, args.foldtype)
        if fold == 0:
            data_metrics(data, log)

        log.info('\n\n --- Fold %d --- ' % (fold + 1))

        for i in xrange(args.repeats):
            #graph = CNN(name='CNN', group=args.groups[0].upper(), data=data)
            graph = WideBoostingNetwork(name='WBN', group=args.groups[0].upper(), data=data)

            graph.train(args, ender=ender, log=log)
            graph.set_meta(fold, 5)

            acc = get_accuracy(graph.meta, 'test', args.symmetry)
            conf, _ = get_confusion_matrix(graph.meta, 'test', args.symmetry)
            log.result('Test set accuracy iteration %d: %.3f' % (i, acc))
            #log.result('Confusion matrix: \n ' + str(conf))

            fold_accuracies.append(acc)
            fold_confs += conf

        log.result('Fold average accuracy: %.3f (~ %.4f)' % (np.mean(fold_accuracies), np.std(fold_accuracies)))
        log.result('Confusion matrix: \n ' + str(fold_confs))
        final_accuracies += fold_accuracies
        final_confs += fold_confs


    log.info('\n\n --- END RESULT ---')
    log.result('Total accuracy: %.3f (~ %.4f)' % (np.mean(final_accuracies), np.std(final_accuracies)))
    log.result('Total confusion matrix: \n' + str(final_confs))

    log.finalize(ender.terminate, False)


#

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
    parser.add_argument("--save_fraction", nargs="?", type=float, default=.5)
    # data arguments
    parser.add_argument("--train", nargs="?", type=str, default='lidc')
    parser.add_argument("--val", nargs="?", type=str, default='empty')
    parser.add_argument("--test", nargs="?", type=str, default="lidc")
    #parser.add_argument("--samples", nargs="+", default=[5000])
    parser.add_argument("--shape", nargs="+", default=[6, 40, 40])

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--symmetry", action="store_true")


    parser.add_argument("--foldtype", nargs="?", type=str, default="3-class", choices=["3-class", "solid", "ggo"])
    parser.add_argument("--folds", nargs="?", type=int, default=5)


    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=0.0001)

    args = parser.parse_args()
    print(args)

    main(args)