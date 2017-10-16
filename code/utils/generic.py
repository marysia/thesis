import datetime
import time
import os
import csv
import pickle
import json
import numpy as np
import sys

sys.path.append('..')
from config import RESULTSDIR, DATADIR

lidc_folder = os.path.join(DATADIR, 'patches', 'lidc-localization-patches')

def log_time(log, args, models):
    start_time = time.time() + (2 * 3600)  # system time is two hours behind; adjust.
    start_str = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    log.info('Starting training at: \t %s' % start_str)

    if args.mode == 'time':
        end_time = start_time + ((args.mode_param * 60) * len(models.keys()) * len(args.groups))
        end_str = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        log.info('Estimated end time: \t %s' % end_str)

def save_pickle(dict, fname):
    with open(fname, 'wb') as f:
        pickle.dump(dict, f)

def save_json(dict, fname):
    with open(fname, 'wb') as f:
        json.dump(dict, f)

def save_meta(dict):
    if not dict == {}:
        elements = [dict['log-identifier'], dict['name'], dict['group'], str(dict['training-set-samples'])]
        fname = os.path.join(RESULTSDIR, 'pickles', '-'.join(elements) + '.pkl')
        print(fname)
        with open(fname, 'wb') as f:
            pickle.dump(dict, f)

def create_submission(graphmeta, symmetry):
    if graphmeta['test-dataset'] != 'lidc-localization':
        return

    if 'test-symmetry-predictions' in graphmeta and symmetry:
        predictions = graphmeta['test-symmetry-predictions']
    else:
        predictions = graphmeta['test-predictions']

    positive = np.load(os.path.join(lidc_folder, 'positive_patches.npz'))
    negative = np.load(os.path.join(lidc_folder, 'negative_patches.npz'))
    positive_meta = positive['meta']
    negative_meta = negative['meta']
    meta = np.concatenate([positive_meta, negative_meta])


    nodules = [['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']]
    for i in xrange(graphmeta['test-set-samples']):
        original_idx = graphmeta['permutation-test-set'][i]
        z, y, x = np.round(meta[original_idx]['center'])
        probability = predictions[i][1]
        seriesuid = str(meta[original_idx]['scan-id'])

        nodules.append([seriesuid, x, y, z, probability])

    name = '-'.join([graphmeta['log-identifier'], graphmeta['name'], graphmeta['group'], str(graphmeta['training-set-samples'])])
    fname = os.path.join(RESULTSDIR, 'submissions', '%s.csv' % name)
    with open(fname, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(nodules)

def submission(test_scope, testdata, predictions, name, log):
    '''
    Args:
        test_scope: string, lidc-localization, nlst-unbalanced, etc.
        testdata: instance of Data class (x, y, id)
        predictions: list of lists
    '''
    if test_scope != 'lidc-localization':
        return

    positive = np.load(os.path.join(lidc_folder, 'positive_patches.npz'))
    negative = np.load(os.path.join(lidc_folder, 'negative_patches.npz'))
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

    fname = os.path.join(RESULTSDIR, 'submissions', '%s.csv' % name)
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