import pickle
import sys
import os
import numpy as np
import glob

from utils.config import RESULTSDIR

def read_pickle(fname):
    with open(fname, 'r') as f:
        meta = pickle.load(f)

    for key, value in meta.items():

        if not (type(value) == list or type(value) == np.array) or len(value) < 10:
            print(key, value)

if __name__ == "__main__":
    files = glob.glob(os.path.join(RESULTSDIR, 'pickles', sys.argv[1] + '*'))

    for f in files:
        read_pickle(f)