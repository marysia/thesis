import glob
import os

import matplotlib as mpl
import pandas as pd

mpl.use('Agg')
import pylab as plt
import numpy as np

results_folder = '/home/marysia/thesis/results'


def sensitivity(df, total_nodules):
    ''' Return sensitivity: tp / (tp + tn)'''
    tp = df[(df.predicted_class == 1) & (df.actual_class == 1)]
    fn = df[(df.predicted_class == 0) & (df.actual_class == 1)]
    if len(tp) == 0 or total_nodules == 0:
        return 0
    return len(tp) / float(total_nodules)


def tp_fp_ratio(df):
    tp = len(df[(df.predicted_class == 1) & (df.actual_class == 1)])
    fp = len(df[(df.predicted_class == 1) & (df.actual_class == 0)])
    return fp / float(tp) if fp > 0 else 0


def score_fp(values, fp_rate, orig_x):
    li = []
    for fp in orig_x:
        idx = np.abs(fp_rate - fp).argmin()
        li.append(values[idx][1])
    return np.mean(np.array(li))


def create_graph(values, id, modelname):
    fp_rate = [elem[0] for elem in values]
    sens = [elem[1] for elem in values]

    # set axis
    orig_x = [0.125, 0.25, 0.5, 1, 2, 4, 8]  # all false positive rates
    orig_y = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # all possible sensitivities

    x = [elem for elem in orig_x if elem < np.max(fp_rate) and elem > np.min(fp_rate)]  # cap x axis range
    y = [elem for elem in orig_y if elem < np.max(sens) and elem > np.min(sens)]  # cap y axis range

    score = score_fp(values, np.array(fp_rate), orig_x)

    plt.plot(fp_rate, sens, label='sensitivity')
    plt.xticks(x)
    plt.yticks(y)
    plt.legend(loc='best')
    plt.title('Simple FROC curve -- %.3f' % score)
    plt.savefig(os.path.join(results_folder, 'plots', 'froc_%s-%s.png' % (id, modelname)))


def get_values(fname):
    # determine model name and log identifier based on filename
    splitted = fname.split('/')[-1].split('-')
    id, modelname = splitted[0], splitted[1]

    df = pd.read_csv(fname)
    total_nodules = len(df[df.actual_class == 1])

    thresholds = np.array(range(0, 100, 1)) / 100.
    values = []
    for threshold in thresholds:
        sub_df = df[df.probability >= threshold]
        # acc = accuracy(sub_df)
        sens = sensitivity(sub_df, total_nodules)
        fp_ratio = tp_fp_ratio(sub_df)

        values.append((fp_ratio, sens))

    values.sort()

    create_graph(values, id, modelname)


fnames = glob.glob(os.path.join(results_folder, 'submissions', '*-submission.csv'))
for fname in fnames:
    get_values(fname)
