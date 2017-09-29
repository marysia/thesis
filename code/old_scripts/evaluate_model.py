"""
Purpose of this script:
    Read in a model (parameter to directory) and apply this model on a (test)dataset (parameter)

    Required:
        - Read in model
        - Apply on dataset.

    NOTE: fp rate is false positives per scan!
"""
import argparse
import glob
import os

import numpy as np
import tensorflow as tf

import models.util.helpers
from models.model_3d import Z3CNN
from preprocessing.patches import DataPatches

models_dict = {
    'Z3CNN': Z3CNN
}


def return_results(modelpath, data):
    checkpoint_file = tf.train.latest_checkpoint(modelpath)
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # build corresponding graph
            graph = get_graph(modelpath, data)

            init = tf.global_variables_initializer()
            sess.run(init)

            # get results
            results = models.util.metrics.get_predictions(sess, graph, data.test.x)
            probabilities = get_probabilities(results)
            return results, probabilities


def get_probabilities(result):
    result = np.array(result)
    differences = [abs(r[0] - r[1]) for r in result]
    max_diff, min_diff = max(differences), min(differences)
    probabilities = [0] * len(differences)
    for i, r in enumerate(result):
        prob = (differences[i] - min_diff) / (max_diff - min_diff)
        if np.argmax(r) == 1:
            probabilities[i] = prob
        else:
            probabilities[i] = 1 - prob
    return probabilities


def get_graph(modelpath, data):
    print('Retrieving graph.')
    graph_name = modelpath.split('/')[-1].split('_')[0]
    print(graph_name)
    model = models_dict[graph_name](model_name='tmp', data=data, log=None, ender=None)
    model.build_model()
    return model


def load_dataset(args):
    return DataPatches(args)


def evaluate_ensemble(ensemble_folder, data):
    model_paths = glob.glob(os.path.join(ensemble_folder, '*'))
    probs = []
    results = []
    for model_path in model_paths:
        print('Evaluating ' + model_path)
        r, p = return_results(model_path, data)
        results.append(r)
        probs.append(p)

    print('Reach!')


if __name__ == "__main__":
    print('Start')
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument("--data", nargs="?", type=str, default="patches3d", choices=['patches2d', 'patches3d', 'mnist'])
    parser.add_argument("--train", nargs="?", type=str, default='nlst-balanced')
    parser.add_argument("--val", nargs="?", type=str, default=None)
    parser.add_argument("--test", nargs="?", type=str, default="lidc-localization")
    parser.add_argument("--shape", nargs="?", type=tuple, default=(8, 30, 30))

    args = parser.parse_args()
    print('loading data')
    data = load_dataset(args)

    ensemble_folder = '/home/marysia/thesis/results/models/'
    evaluate_ensemble(ensemble_folder, data)
