"""
Purpose of this script:
    Read in a model (parameter to directory) and apply this model on a (test)dataset (parameter)

    Required:
        - Read in model
        - Apply on dataset.

    NOTE: fp rate is false positives per scan!
"""
import tensorflow as tf
from preprocessing.lidc import LIDCTestPatches
from preprocessing.patches import DataPatches
modelpath = '/home/marysia/thesis/results/models/'



def load_model():
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, modelpath)

    pass

def load_dataset():
    return DataPatches(shape=(8, 30, 30), small=True)
    #return LIDCTestPatches(shape=(8, 30, 30))


def apply_model():
    pass