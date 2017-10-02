import datetime
import time
import os
import csv
import numpy as np

results_folder = '/home/marysia/thesis/results/'

def log_time(log, args, models):
    start_time = time.time() + (2 * 3600)  # system time is two hours behind; adjust.
    start_str = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    log.info('Starting training at: \t %s' % start_str)

    if args.mode == 'time':
        end_time = start_time + ((args.mode_param * 60) * len(models.keys()))
        end_str = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        log.info('Estimated end time: \t %s' % end_str)

def submission(test_scope, testdata, predictions, name, log):
    '''
    Args:
        test_scope: string, lidc-localization, nlst-unbalanced, etc.
        testdata: instance of Data class (x, y, id)
        predictions: list of lists
    '''
    if test_scope != 'lidc-localization':
        return

    positive = np.load('/home/marysia/data/thesis/patches/lidc-localization-patches/positive_patches.npz')
    negative = np.load('/home/marysia/data/thesis/patches/lidc-localization-patches/negative_patches.npz')
    positive_meta = positive['meta']
    negative_meta = negative['meta']
    meta = np.concatenate([positive_meta, negative_meta])

    nodules = [['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']]

    for i in xrange(testdata.samples):
        original_idx = testdata.id[i]
        z, y, x = np.round(meta[original_idx]['center'], 1)
        probability = predictions[i][1]
        seriesuid = str(meta[original_idx]['scan-id'])

        nodules.append([seriesuid, x, y, z, probability])

    fname = os.path.join(results_folder, 'submissions', '%s.csv' % name)
    with open(fname, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(nodules)

def get_best_predictions(model_evaluations, ensemble):
    ''' Returns predictions. Either best predictions, based on a
    trade-off between sensitivity and false-positive rates, or the
    combined predictions of all models.'''

    if ensemble:
        results = [elem['results'] for elem in model_evaluations]
        w = [elem['weight'] for elem in model_evaluations]

        predictions = np.zeros_like(results[0])
        for i in xrange(len(predictions)):
            tmp = np.array(results[0][i]) * w[0]    # prediction of model * weight of model
            for j in xrange(1, len(results)):
                tmp += np.array(results[j][i]) * w[j]   # prediction of model * weight of model
            tmp = tmp / sum(w)  # / sum of all weights to sum to 1.
            predictions[i] = tmp
    else:
        fp = [elem['fp-rate'] for elem in model_evaluations]
        sens = [elem['sensitivity'] for elem in model_evaluations]

        sorted_fp = sorted(fp)
        sorted_sens = sorted(sens, reverse=True)

        scores_fp = [sorted_fp.index(elem) for elem in fp]
        scores_sens = [sorted_sens.index(elem) for elem in sens]
        combined_scores = np.array(scores_fp) + np.array(scores_sens)
        idx = np.argmin(combined_scores)
        predictions = model_evaluations[idx]['results']

    return predictions