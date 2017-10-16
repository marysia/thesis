import os
import math
import sys
import glob
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,LogFormatter,StrMethodFormatter,FixedFormatter

import sklearn.metrics as skl_metrics
import numpy as np
import argparse
import re

from utils.config import HOME, RESULTSDIR
luna_evaluation_dir = os.path.join(HOME, 'thesis', 'luna_evaluation')
sys.path.append(luna_evaluation_dir)
from NoduleFinding import NoduleFinding

from tools import csvTools

# Evaluation settings
bPerformBootstrapping = False
bNumberOfBootstrapSamples = 1000
bOtherNodulesAsIrrelevant = True
bConfidence = 0.95

seriesuid_label = 'seriesuid'
coordX_label = 'coordX'
coordY_label = 'coordY'
coordZ_label = 'coordZ'
diameter_mm_label = 'diameter_mm'
CADProbability_label = 'probability'

# plot settings
FROC_minX = 0.125 # Mininum value of x-axis of FROC curve
FROC_maxX = 8 # Maximum value of x-axis of FROC curve
bLogPlot = True

def generateBootstrapSet(scanToCandidatesDict, FROCImList):
    '''
    Generates bootstrapped version of set
    '''
    imageLen = FROCImList.shape[0]
    
    # get a random list of images using sampling with replacement
    rand_index_im   = np.random.randint(imageLen, size=imageLen)
    FROCImList_rand = FROCImList[rand_index_im]
    
    # get a new list of candidates
    candidatesExists = False
    for im in FROCImList_rand:
        if im not in scanToCandidatesDict:
            continue
        
        if not candidatesExists:
            candidates = np.copy(scanToCandidatesDict[im])
            candidatesExists = True
        else:
            candidates = np.concatenate((candidates,scanToCandidatesDict[im]),axis = 1)

    return candidates

def compute_mean_ci(interp_sens, confidence = 0.95):
    sens_mean = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    sens_lb   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    sens_up   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    
    Pz = (1.0-confidence)/2.0
        
    for i in range(interp_sens.shape[1]):
        # get sorted vector
        vec = interp_sens[:,i]
        vec.sort()

        sens_mean[i] = np.average(vec)
        sens_lb[i] = vec[math.floor(Pz*len(vec))]
        sens_up[i] = vec[math.floor((1.0-Pz)*len(vec))]

    return sens_mean,sens_lb,sens_up

def computeFROC_bootstrap(FROCGTList,FROCProbList,FPDivisorList,FROCImList,excludeList,numberOfBootstrapSamples=1000, confidence = 0.95):

    set1 = np.concatenate(([FROCGTList], [FROCProbList], [excludeList]), axis=0)
    
    fps_lists = []
    sens_lists = []
    thresholds_lists = []
    
    FPDivisorList_np = np.asarray(FPDivisorList)
    FROCImList_np = np.asarray(FROCImList)
    
    # Make a dict with all candidates of all scans
    scanToCandidatesDict = {}
    for i in range(len(FPDivisorList_np)):
        seriesuid = FPDivisorList_np[i]
        candidate = set1[:,i:i+1]

        if seriesuid not in scanToCandidatesDict:
            scanToCandidatesDict[seriesuid] = np.copy(candidate)
        else:
            scanToCandidatesDict[seriesuid] = np.concatenate((scanToCandidatesDict[seriesuid],candidate),axis = 1)

    for i in range(numberOfBootstrapSamples):
        print 'computing FROC: bootstrap %d/%d' % (i,numberOfBootstrapSamples)
        # Generate a bootstrapped set
        btpsamp = generateBootstrapSet(scanToCandidatesDict,FROCImList_np)
        fps, sens, thresholds = computeFROC(btpsamp[0,:],btpsamp[1,:],len(FROCImList_np),btpsamp[2,:])
    
        fps_lists.append(fps)
        sens_lists.append(sens)
        thresholds_lists.append(thresholds)

    # compute statistic
    all_fps = np.linspace(FROC_minX, FROC_maxX, num=10000)
    
    # Then interpolate all FROC curves at this points
    interp_sens = np.zeros((numberOfBootstrapSamples,len(all_fps)), dtype = 'float32')
    for i in range(numberOfBootstrapSamples):
        interp_sens[i,:] = np.interp(all_fps, fps_lists[i], sens_lists[i])
    
    # compute mean and CI
    sens_mean,sens_lb,sens_up = compute_mean_ci(interp_sens, confidence = confidence)

    return all_fps, sens_mean, sens_lb, sens_up

def computeFROC(FROCGTList, FROCProbList, totalNumberOfImages, excludeList):
    # Remove excluded candidates
    FROCGTList_local = []
    FROCProbList_local = []
    for i in range(len(excludeList)):
        if excludeList[i] == False:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])
    
    numberOfDetectedLesions = sum(FROCGTList_local)
    totalNumberOfLesions = sum(FROCGTList)
    totalNumberOfCandidates = len(FROCProbList_local)
    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)
    if sum(FROCGTList) == len(FROCGTList): # Handle border case when there are no false positives and ROC analysis give nan values.
      print "WARNING, this system has no false positives.."
      fps = np.zeros(len(fpr))
    else:
      fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages
    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
    return fps, sens, thresholds

def evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules, CADSystemName, maxNumberOfCADMarks=-1,
                performBootstrapping=False,numberOfBootstrapSamples=1000,confidence = 0.95):
    '''
    function to evaluate a CAD algorithm
    @param seriesUIDs: list of the seriesUIDs of the cases to be processed
    @param results_filename: file with results
    @param outputDir: output directory
    @param allNodules: dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
    @param CADSystemName: name of the CAD system, to be used in filenames and on FROC curve
    '''

    results = csvTools.readCSV(results_filename)

    allCandsCAD = {}
    
    for seriesuid in seriesUIDs:
        
        # collect candidates from result file
        nodules = {}
        header = results[0]
        
        i = 0
        for result in results[1:]:
            nodule_seriesuid = result[header.index(seriesuid_label)]
            
            if seriesuid == nodule_seriesuid:
                nodule = getNodule(result, header)
                nodule.candidateID = i
                nodules[nodule.candidateID] = nodule
                i += 1

        if (maxNumberOfCADMarks > 0):
            # number of CAD marks, only keep must suspicous marks

            if len(nodules.keys()) > maxNumberOfCADMarks:
                # make a list of all probabilities
                probs = []
                for keytemp, noduletemp in nodules.iteritems():
                    probs.append(float(noduletemp.CADprobability))
                probs.sort(reverse=True) # sort from large to small
                probThreshold = probs[maxNumberOfCADMarks]
                nodules2 = {}
                nrNodules2 = 0
                for keytemp, noduletemp in nodules.iteritems():
                    if nrNodules2 >= maxNumberOfCADMarks:
                        break
                    if float(noduletemp.CADprobability) > probThreshold:
                        nodules2[keytemp] = noduletemp
                        nrNodules2 += 1

                nodules = nodules2
        
        #print 'adding candidates: ' + seriesuid
        allCandsCAD[seriesuid] = nodules
    
    # open output files
    #nodNoCandFile = open(os.path.join(outputDir, "nodulesWithoutCandidate_%s.txt" % CADSystemName), 'w')
    
    # --- iterate over all cases (seriesUIDs) and determine how
    # often a nodule annotation is not covered by a candidate

    # initialize some variables to be used in the loop
    candTPs = 0
    candFPs = 0
    candFNs = 0
    candTNs = 0
    totalNumberOfCands = 0
    totalNumberOfNodules = 0
    doubleCandidatesIgnored = 0
    irrelevantCandidates = 0
    minProbValue = -1000000000.0 # minimum value of a float
    FROCGTList = []
    FROCProbList = []
    FPDivisorList = []
    excludeList = []
    FROCtoNoduleMap = []
    ignoredCADMarksList = []

    # -- loop over the cases
    for seriesuid in seriesUIDs:
        # get the candidates for this case
        try:
            candidates = allCandsCAD[seriesuid]
        except KeyError:
            candidates = {}

        # add to the total number of candidates
        totalNumberOfCands += len(candidates.keys())

        # make a copy in which items will be deleted
        candidates2 = candidates.copy()

        # get the nodule annotations on this case
        try:
            noduleAnnots = allNodules[seriesuid]
        except KeyError:
            noduleAnnots = []

        # - loop over the nodule annotations
        for noduleAnnot in noduleAnnots:
            # increment the number of nodules
            if noduleAnnot.state == "Included":
                totalNumberOfNodules += 1

            x = float(noduleAnnot.coordX)
            y = float(noduleAnnot.coordY)
            z = float(noduleAnnot.coordZ)

            # 2. Check if the nodule annotation is covered by a candidate
            # A nodule is marked as detected when the center of mass of the candidate is within a distance R of
            # the center of the nodule. In order to ensure that the CAD mark is displayed within the nodule on the
            # CT scan, we set R to be the radius of the nodule size.
            diameter = float(noduleAnnot.diameter_mm)
            if diameter < 0.0:
              diameter = 10.0
            radiusSquared = pow((diameter / 2.0), 2.0)

            found = False
            noduleMatches = []
            for key, candidate in candidates.iteritems():
                x2 = float(candidate.coordX)
                y2 = float(candidate.coordY)
                z2 = float(candidate.coordZ)
                dist = math.pow(x - x2, 2.) + math.pow(y - y2, 2.) + math.pow(z - z2, 2.)
                if dist < radiusSquared:
                    if (noduleAnnot.state == "Included"):
                        found = True
                        noduleMatches.append(candidate)
                        if key not in candidates2.keys():
                            print "This is strange: CAD mark %s detected two nodules! Check for overlapping nodule annotations, SeriesUID: %s, nodule Annot ID: %s" % (str(candidate.id), seriesuid, str(noduleAnnot.id))
                        else:
                            del candidates2[key]
                    elif (noduleAnnot.state == "Excluded"): # an excluded nodule
                        if bOtherNodulesAsIrrelevant: #    delete marks on excluded nodules so they don't count as false positives
                            if key in candidates2.keys():
                                irrelevantCandidates += 1
                                ignoredCADMarksList.append("%s,%s,%s,%s,%s,%s,%.9f" % (seriesuid, -1, candidate.coordX, candidate.coordY, candidate.coordZ, str(candidate.id), float(candidate.CADprobability)))
                                del candidates2[key]
            if len(noduleMatches) > 1: # double detection
                doubleCandidatesIgnored += (len(noduleMatches) - 1)
            if noduleAnnot.state == "Included":
                # only include it for FROC analysis if it is included
                # otherwise, the candidate will not be counted as FP, but ignored in the
                # analysis since it has been deleted from the nodules2 vector of candidates
                if found == True:
                    # append the sample with the highest probability for the FROC analysis
                    maxProb = None
                    for idx in range(len(noduleMatches)):
                        candidate = noduleMatches[idx]
                        if (maxProb is None) or (float(candidate.CADprobability) > maxProb):
                            maxProb = float(candidate.CADprobability)

                    FROCGTList.append(1.0)
                    FROCProbList.append(float(maxProb))
                    FPDivisorList.append(seriesuid)
                    excludeList.append(False)
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%s,%.9f" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), str(candidate.id), float(candidate.CADprobability)))
                    candTPs += 1
                else:
                    candFNs += 1
                    # append a positive sample with the lowest probability, such that this is added in the FROC analysis
                    FROCGTList.append(1.0)
                    FROCProbList.append(minProbValue)
                    FPDivisorList.append(seriesuid)
                    excludeList.append(True)
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%s,%s" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), int(-1), "NA"))
                    #nodNoCandFile.write("%s,%s,%s,%s,%s,%.9f,%s\n" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), str(-1)))

        # add all false positives to the vectors
        for key, candidate3 in candidates2.iteritems():
            candFPs += 1
            FROCGTList.append(0.0)
            FROCProbList.append(float(candidate3.CADprobability))
            FPDivisorList.append(seriesuid)
            excludeList.append(False)
            FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%s,%.9f" % (seriesuid, -1, candidate3.coordX, candidate3.coordY, candidate3.coordZ, str(candidate3.id), float(candidate3.CADprobability)))

    # compute FROC
    fps, sens, thresholds = computeFROC(FROCGTList,FROCProbList,len(seriesUIDs),excludeList)

    #calculate overall score:
    fp_points = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    li = []
    for point in fp_points:
        idx = np.abs(fps - point).argmin()
        li.append(sens[idx])
    overall_score = np.mean(np.array(li))


    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)
    sens_itp = np.interp(fps_itp, fps, sens)

    return sens_itp, overall_score

    
def getNodule(annotation, header, state = ""):
    nodule = NoduleFinding()
    nodule.coordX = annotation[header.index(coordX_label)]
    nodule.coordY = annotation[header.index(coordY_label)]
    nodule.coordZ = annotation[header.index(coordZ_label)]
    
    if diameter_mm_label in header:
        nodule.diameter_mm = annotation[header.index(diameter_mm_label)]
    
    if CADProbability_label in header:
        nodule.CADprobability = annotation[header.index(CADProbability_label)]
    
    if not state == "":
        nodule.state = state

    return nodule
    
def collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs):
    allNodules = {}
    noduleCount = 0
    noduleCountTotal = 0
    
    for seriesuid in seriesUIDs:
        print 'adding nodule annotations: ' + seriesuid
        
        nodules = []
        numberOfIncludedNodules = 0
        
        # add included findings
        header = annotations[0]
        for annotation in annotations[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]
            
            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state = "Included")
                nodules.append(nodule)
                numberOfIncludedNodules += 1
        
        # add excluded findings
        header = annotations_excluded[0]
        for annotation in annotations_excluded[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]
            
            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state = "Excluded")
                nodules.append(nodule)
            
        allNodules[seriesuid] = nodules
        noduleCount      += numberOfIncludedNodules
        noduleCountTotal += len(nodules)
    
    print 'Total number of included nodule annotations: ' + str(noduleCount)
    print 'Total number of nodule annotations: ' + str(noduleCountTotal)
    return allNodules
    
    
def collect(annotations_filename,annotations_excluded_filename,seriesuids_filename):
    annotations          = csvTools.readCSV(annotations_filename)
    annotations_excluded = csvTools.readCSV(annotations_excluded_filename)
    seriesUIDs_csv = csvTools.readCSV(seriesuids_filename)
    
    seriesUIDs = []
    for seriesUID in seriesUIDs_csv:
        seriesUIDs.append(seriesUID[0])

    allNodules = collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs)
    
    return (allNodules, seriesUIDs)
    
    
def noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,outputDir):
    '''
    function to load annotations and evaluate a CAD algorithm
    @param annotations_filename: list of annotations
    @param annotations_excluded_filename: list of annotations that are excluded from analysis
    @param seriesuids_filename: list of CT images in seriesuids
    @param results_filename: list of CAD marks with probabilities
    @param outputDir: output directory
    '''
    
    print annotations_filename
    
    #(allNodules, seriesUIDs) = collect(annotations_filename, annotations_excluded_filename, seriesuids_filename)
    with open(os.path.join(luna_evaluation_dir, 'allNodules.pkl'), 'rb') as f:
        allNodules = pickle.load(f)
    with open(os.path.join(luna_evaluation_dir, 'allNodules.pkl'), 'rb') as f:
        seriesUIDs = pickle.load(f)

    evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules,
                os.path.splitext(os.path.basename(results_filename))[0],
                maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)



def create_froc_curve(result_files, fname):
    outputDir = os.path.join(RESULTSDIR, 'CADEvaluation')

    with open(os.path.join(luna_evaluation_dir, 'allNodules.pkl'), 'rb') as f:
        allNodules = pickle.load(f)
    with open(os.path.join(luna_evaluation_dir, 'allNodules.pkl'), 'rb') as f:
        seriesUIDs = pickle.load(f)


    ax = plt.gca()
    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)
    for results_filename in result_files:
        # execute only if run as a script
        CADSystemName = os.path.splitext(os.path.basename(results_filename))[0]
        sens_itp, overall_score = evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules,
                                    CADSystemName,
                                    maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                                    numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)
        plt.plot(fps_itp, sens_itp, label="(%.3f) \t %s" % (overall_score, CADSystemName), lw=2)


    xmin = FROC_minX
    xmax = FROC_maxX
    plt.xlim(xmin, xmax)
    plt.ylim(0, 1)
    plt.xlabel('Average number of false positives per scan')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.title('FROC performance')

    if bLogPlot:
        plt.xscale('log', basex=2)
        ax.xaxis.set_major_formatter(FixedFormatter([0.125, 0.25, 0.5, 1, 2, 4, 8]))

    # set your ticks manually
    ax.xaxis.set_ticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    plt.grid(b=True, which='both')
    plt.tight_layout()

    plt.savefig(os.path.join(outputDir, "%s.png" % fname), bbox_inches=0, dpi=300)

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def get_files(files, constraints):
    negative_files = []
    for f in files:
        file = f.split('/')[-1]
        for c in constraints:
            try:
                int(c)
                if not '-' + c + '.csv' in file.lower():
                    negative_files.append(f)
            except:
                if not c.lower() in file.lower():
                    negative_files.append(f)
    files = list(set(files) - set(negative_files))
    files.sort(key=natural_keys)
    return files

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--constraints", nargs="+", default=[])
    parser.add_argument("--fname", nargs="?", type=str, default='froc')
    parser.add_argument("--final", action="store_true")     # in the final_submissions folder instead of submissions

    args = parser.parse_args()

    if args.final:
        all_files = glob.glob(os.path.join(RESULTSDIR, 'final_submissions', '*.csv'))
    else:
        all_files = glob.glob(os.path.join(RESULTSDIR, 'submissions', '*.csv'))

    files = get_files(all_files, args.constraints)

    create_froc_curve(files, args.fname)
