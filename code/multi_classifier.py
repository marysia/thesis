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

def evaluate_testset(ggo, solid, data, log):
    indiv_classifiers = {'solid': 0, 'ggo': 0}
    three_class = np.zeros(shape=(3,4))
    for i in xrange(data.test.samples):
        idx_solid = solid.meta['permutation-test-set'][i]
        idx_ggo = ggo.meta['permutation-test-set'][i]

        actually_solid = solid.meta['labels-test-set'][idx_solid]
        actually_ggo = ggo.meta['labels-test-set'][idx_ggo]

        prediction_solid = np.argmax(solid.meta['test-predictions'][idx_solid])
        prediction_ggo = np.argmax(ggo.meta['test-predictions'][idx_ggo])

        if actually_solid == prediction_solid:
            indiv_classifiers['solid'] += 1
        if actually_ggo == prediction_ggo:
            indiv_classifiers['ggo'] += 1


        if actually_solid  and not actually_ggo:
            real_idx = 0
        elif actually_solid and actually_ggo:
            real_idx = 1
        elif not actually_solid and actually_ggo:
            real_idx = 2

        if prediction_solid and not prediction_ggo:
            pred_idx = 0
        elif prediction_solid and prediction_ggo:
            pred_idx = 1
        elif not prediction_solid and prediction_ggo:
            pred_idx = 2
        else:
            pred_idx = 3

        three_class[real_idx, pred_idx] += 1

    log.result(indiv_classifiers)
    log.result(three_class)
    acc = (three_class[0,0] + three_class[1,1] + three_class[2,2]) / np.sum(three_class)
    log.result('Accuracy: ' + str(acc))
    return three_class

def get_acc(data, graph):
    correct = 0
    for i in xrange(data.test.samples):
        if np.argmax(graph.meta['test-predictions'][i]) == graph.meta['labels-test-set'][i]:
            correct += 1

    return correct / float(data.test.samples)

def main(args):
    data = DataComposition(args)
    log = Logger(args, LOGDIR, logfolder='composition', logname=args.log)

    solid_acc = []
    ggo_acc = []
    three_class = np.zeros(shape=(3,4))


    for fold in xrange(args.folds):
    # get classification for solid
        print 'Training Solid'
        data.set_fold(fold, 'solid')
        solidgraph = WideBoostingNetwork(name='WBN', group=args.groups[0].upper(), data=data)
        #solidgraph = CNN(name='CNN', group=args.groups[0].upper(), data=data)
        solidgraph.train(args, ender=ender, log=log)
        solidgraph.set_meta(0, 5)

        solid_acc.append(get_acc(data, solidgraph))
        log.result('Solid accuracy: ' + str(get_acc(data, solidgraph)))
        # get classification for ggo
        print 'Training GGO'
        data.set_fold(fold, 'ggo')
        ggograph = WideBoostingNetwork(name='WBN', group=args.groups[0].upper(), data=data)
        #ggograph = CNN(name='CNN', group=args.groups[0].upper(), data=data)
        ggograph.train(args, ender=ender, log=log)
        ggograph.set_meta(0, 5)

        ggo_acc.append(get_acc(data, ggograph))
        log.result('GGO accuracy: ' + str(get_acc(data, ggograph)))

        conf = evaluate_testset(ggograph, solidgraph, data, log)
        three_class += conf


    log.result('Solid accuracies: ' + str(solid_acc))
    log.result('GGO accuracies: ' + str(ggo_acc))
    acc = (three_class[0,0] + three_class[1,1] + three_class[2,2]) / np.sum(three_class)
    log.result('Overall confusion matrix: \n')
    log.result(three_class)
    log.result('Overall accuracy: ' + str(acc))
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