'''
Categories:
    - Subtlety: subtlety of detection
    - Structure: internal composition of the nodule
    - Calcification: pattern of calcification, if present
    - Sphericity: 3D shape of nodule in terms of roundness
    - Margin: description of how well defined the margin is
    - Lobulation: degree of lobulation ranging from None to marked
    - Spiculation: extend of spiculation present
    - Texture: ggo, solid, mixed
    - Malignancy: subjective assessment of malignancy, assuming the scan originated from 60 year old smoker

    All on five-point scale, except internal structure and calcification
'''
import os
import numpy as np
import pandas as pd
from utils.config import DATADIR, RESULTSDIR
import glob

categories = ['subtlety', 'structure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']

def analyse_lidc():
    fn = os.path.join(DATADIR, 'patches', 'lidc-localization-patches', 'positive_patches.npz')
    meta = np.load(fn)['meta']
    annotations = [elem['annotation-metadata'] for elem in meta]

    for c in categories:
        values = np.array([np.median(elem[c]) for elem in annotations])
        print('%s \t min: %.2f \t max %.2f \t mean: %.2f \t std: %.4f' % (c, np.min(values), np.max(values), np.mean(values), np.std(values)))
        v = list(values)
        print('\t \t tot: %d 1: %d 3: %d 5: %d\n' % (len(v), v.count(1), v.count(3), v.count(5))  )


def create_dict():
    fn = os.path.join(DATADIR, 'patches', 'lidc-localization-patches', 'positive_patches.npz')
    meta = np.load(fn)['meta']
    d = {}
    for nodule in meta:
        id = nodule['scan-id']
        nodule['annotation-metadata']['real-center'] = nodule['center']
        if id in d.keys():
            d[id].append(nodule['annotation-metadata'])
        else:
            d[id] = [nodule['annotation-metadata']]

    return d

def get_nodule_malignancy(row, d):
    scan_meta = d[row['seriesuid']]
    center = [row['coordZ'], row['coordY'], row['coordX']]
    for nodule in scan_meta:
        nodule_center = list(np.round(nodule['real-center']))
        if nodule_center == center:
            return nodule['malignancy']
    return None

def analyse_submission_malignancy(fname, group):
    headers = [['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']]
    df = pd.read_csv(fname)

    # get malignant scans
    d = create_dict()
    probs = []
    probs_low = []
    probs_medium = []
    missed = 0
    for i, row in df.iterrows():
        if row['seriesuid'] in d.keys():
            center = [row['coordZ'], row['coordY'], row['coordX']]
            nodule_scan = d[row['seriesuid']]

            for elem in nodule_scan:

                nodule_center = list(np.round(elem['real-center']))
                if nodule_center == center:
                    if np.median(elem['malignancy']) >= 4:
                        #print  str(i) + ' Malignant. %s %.5f ' % ( str(elem['malignancy']), row['probability'])
                        probs.append(row['probability'])
                        if row['probability'] < .5:
                            missed += 1
                    elif np.median(elem['malignancy']) < 2:
                        probs_low.append(row['probability'])
                    else:
                        probs_medium.append(row['probability'])

    print '\nGroup ' + group
    print 'Mean and std malignant nodules (%d total): %.3f (%.4f)' % (len(probs), np.mean(probs), np.std(probs))
    print 'Mean and std benign nodules (%d total): %.3f (%.4f)' % (len(probs_low), np.mean(probs_low), np.std(probs_low))

    total = probs + probs_low + probs_medium
    print 'Malignant nodules missed at classification: %d' % missed
    #print 'Mean total: ' + str(np.mean(total)) + ', missed nodules: ' + str(missed)
    #print 'Lowest malignant probability: ' + str(np.min(probs))


def analyse_submission_top_percentage(fn, group, percentage):

    df = pd.read_csv(fn)
    nr = int(round(percentage * len(df)))
    subd = df.nlargest(len(df), 'probability')
    d = create_dict()
    tot = []
    total_benign = 0
    total_benign_found = 0

    total_malignant_nodules = 0
    total_malignant_nodules_found = 0
    nodules = 0

    for i, row in subd.iterrows():
        if row['seriesuid'] in d.keys():
            if i < nr:
                nodules += 1
            malignancy = get_nodule_malignancy(row, d)
            if malignancy is not None:

                median_malignancy = np.median(malignancy)

                malignancy.sort()

                m = True if len(malignancy) >= 3 and malignancy[-3] >= 4 else False
                #if median_malignancy >= 4:
                if m:
                    total_malignant_nodules += 1

                if len(tot) < nr:
                    if m:
                        total_malignant_nodules_found += 1
                    tot.append(median_malignancy)


                b = True if len(malignancy) >= 3 and malignancy[-1] <= 2 else False
                if b:
                    total_benign += 1
                    if len(tot) < nr:
                        total_benign_found += 1

    print 'Group %s: %d of %d' % (group, total_malignant_nodules_found, total_malignant_nodules)
    #print 'Average malignancy for the top %.0f percent of nodules for group %s: %.3f (%.4f)' % ((percentage * 100), group, np.mean(tot), np.std(tot))
    #print 'Total number of malignant nodules in top %.0f percent: %d of %d\n' % ((percentage * 100), total_malignant_nodules_found, total_malignant_nodules)
    #print 'Total number of benign nodules in top %.0f percent: %d of %d' % ((percentage * 100), total_benign_found, total_benign)
    #print 'Total number of nodules in top %.0f percent: %d \n' % ((percentage * 100), nodules)

def is_nodule(row, d):
    if row['seriesuid'] not in d.keys():
        return False
    scan_meta = d[row['seriesuid']]
    center = [row['coordZ'], row['coordY'], row['coordX']]
    for nodule in scan_meta:
        nodule_center = list(np.round(nodule['real-center']))
        if nodule_center == center:
            return True
    return False

def test(fn, percentage=0.1, limit=500):

    # get top 10% probabilities
    # get accuracy on top 10% & total nodules
    # get how many are malignant
    # get percentage of nodules malignant


    df = pd.read_csv(fn)
    nr = int(round(percentage * len(df)))
    subd = df.nlargest(len(df), 'probability')
    d = create_dict()
    counts = {'nodules': 0, 'malignant': 0, 'total': 0}
    for i, row in subd.iterrows():
        counts['total'] += 1
        if is_nodule(row, d):
            counts['nodules'] += 1

        if row['seriesuid'] in d.keys() and counts['nodules'] < limit:
            malignancy = get_nodule_malignancy(row, d)
            if malignancy is not None:
                malignancy.sort()
                m = True if len(malignancy) >= 3 and malignancy[-3] >= 4 else False
                if m:
                    counts['malignant'] += 1

    print fn.split('/')[-1], limit, counts['malignant']



limits = [50, 100, 150, 200]
csvs = glob.glob(os.path.join(RESULTSDIR, 'submissions-best', '*'))
for a in csvs:
    if str(300)+'-' in a and 'True' in a:

            for limit in limits:
                test(a, percentage=.1, limit=limit)




# groups = ['Z3', 'C4h', 'D4h', 'O']
# set_size = [30, 300, 3000, 30000]
#
#
# csvs = glob.glob(os.path.join(RESULTSDIR, 'submissions-best', '*'))
#
# for s in set_size:
#     print 'Size: %d' % s
#     for g in groups:
#         fns = [elem for elem in csvs if g.upper() in elem and str(s)+'-' in elem and 'True' in elem]
#         if len(fns) > 0:
#             fn = fns[0]
#             analyse_submission_top_percentage(fn, g, .1)

# fn = os.path.join(RESULTSDIR, 'submissions', 'ctznsz-CNN-Z3-300-4.csv')
# #analyse_submission_malignancy(fn, 'Z3')
# analyse_submission_top_percentage(fn, 'Z3', .1)
#
# fn = os.path.join(RESULTSDIR, 'submissions', 'ctznsz-CNN-C4H-300-0.csv')
# # analyse_submission_malignancy(fn, 'C4h')
# analyse_submission_top_percentage(fn, 'C4h', .1)
# #
# fn = os.path.join(RESULTSDIR, 'submissions', 'xbhlkw-CNN-D4H-300-0.csv')
# # analyse_submission_malignancy(fn, 'D4h')
# analyse_submission_top_percentage(fn, 'D4h', .1)
# #
# fn = os.path.join(RESULTSDIR, 'submissions', 'htcowc-CNN-O-300-3.csv')
# # analyse_submission_malignancy(fn, 'O')
# analyse_submission_top_percentage(fn, 'O', .1)
# #
# #
# fn = os.path.join(RESULTSDIR, 'submissions', 'gttjgo-CNN-Z3-3000-0.csv')
# # analyse_submission_malignancy(fn, 'Z3')
# analyse_submission_top_percentage(fn, 'Z3', .1)
# #
# fn = os.path.join(RESULTSDIR, 'submissions', 'fusjiz-CNN-C4H-3000-0.csv')
# # analyse_submission_malignancy(fn, 'C4h')
# analyse_submission_top_percentage(fn, 'C4h', .1)
# #
# fn = os.path.join(RESULTSDIR, 'submissions', 'umcvzo-CNN-D4H-3000-0.csv')
# # analyse_submission_malignancy(fn, 'D4h')
# analyse_submission_top_percentage(fn, 'D4h', .1)
# #
# fn = os.path.join(RESULTSDIR, 'submissions', 'elstem-CNN-O-3000-3.csv')
# # analyse_submission_malignancy(fn, 'O')
# analyse_submission_top_percentage(fn, 'O', .1)