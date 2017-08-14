import matplotlib as mpl
mpl.use('Agg')

import glob
import os
import argparse
import pandas as pd
import pylab as plt


results_folder = '/home/marysia/thesis/results'

def accuracy(df):
    ''' Return accuracy: (tp + tn) / (tp + tn + fp + fn)'''
    correct_predictions = df[df.predicted_class == df.actual_class]
    return len(correct_predictions) / float(len(df)) if len(df) > 0 else 0

def sensitivity(df):
    ''' Return sensitivity: tp / (tp + tn)'''
    tp = df[(df.predicted_class == 1) & (df.actual_class == 1)]
    fn = df[(df.predicted_class == 0) & (df.actual_class == 1)]
    return len(tp) / float(len(tp) + len(fn)) if (len(tp)) > 0 or len(fn) > 0 else 0

def tp_fp_ratio(df):
    tp = len(df[(df.predicted_class == 1) & (df.actual_class == 1)])
    fp = len(df[(df.predicted_class == 1) & (df.actual_class == 0)])
    return fp/float(tp) if fp > 0 else 0

def create_plot(modelname, id, t, a, s, fp):
    plt.plot(t, a, label='accuracy')
    plt.plot(t, s, label='sensitivity')
    plt.plot(t, fp, label='fp-ratio')
    plt.xticks(t)
    plt.yticks([.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    plt.legend(loc='best')
    plt.savefig(os.path.join(results_folder, 'plots', '%s-%s.png' % (id, modelname)))

def analyse_file(fname):
    # determine model name and log identifier based on filename
    splitted = fname.split('/')[-1].split('-')
    id, modelname = splitted[0], splitted[1]

    df = pd.read_csv(fname)
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    acc = []
    sens = []
    fp_ratio = []
    print('\n--- Results of %s (%d cases) --- ' % (modelname, len(df)))
    for threshold in thresholds:
        sub_df = df[df.probability >= threshold]
        acc.append(accuracy(sub_df))
        sens.append(sensitivity(sub_df))
        fp_ratio.append(tp_fp_ratio(sub_df))
        print('(%d) \t thr: %.1f \t sens: %.2f \t acc: %.2f \t fp-ratio: %.2f ' % (len(sub_df), threshold, sens[-1], acc[-1], fp_ratio[-1]))

    create_plot(modelname, id, thresholds, acc, sens, fp_ratio)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()
    fpath = os.path.join(results_folder, args.model + '-submission.csv')

    if os.path.exists(fpath):
        print('Analysing %s model submission file.' % args.model)
        analyse_file(fpath)
    else:
        print('Analysing all submission files.')
        for fname in glob.glob(os.path.join(results_folder, '*-submission.csv')):
            analyse_file(fname)