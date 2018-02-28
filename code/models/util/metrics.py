import numpy as np
import tensorflow as tf
from augmentation import augment_dataset
BATCH = 100

def get_batch_size(sess, model, x):
    '''
    Return largest batch size possible for network to perform inference on.
    '''
    batch_size = 512
    while batch_size > 1:
        try:
            feed_dict = {model.x: x[0:batch_size]}
            sess.run(model.model_logits, feed_dict=feed_dict)
            return batch_size
        except:
            batch_size /= 2
    return 1


def get_results(meta, scope, symmetry):
    '''
    Returns predictions and true labels
    '''
    labels = meta['labels-' + scope + '-set']
    predictions_str = scope
    predictions_str += '-symmetry' if symmetry else ''
    predictions_str += '-predictions'
    predictions = meta[predictions_str]
    return predictions, labels

def get_accuracy(meta, scope, symmetry):
    '''
    Returns accuracy
    '''
    predictions, labels = get_results(meta, scope, symmetry)

    correct = 0
    for i in xrange(len(labels)):
        print 'Ground Truth: ', labels[i], ' Probabilities: ', predictions[i]
        if labels[i] == np.argmax(predictions[i]):
            correct += 1
    return correct / float(len(labels))

def get_confusion_matrix(meta, scope, symmetry):
    '''
    Returns confusion matrix
    '''
    predictions, labels = get_results(meta, scope, symmetry)

    nb_classes = predictions[0].shape[0]

    confusion_matrix = np.zeros(shape=(nb_classes, nb_classes))

    for i, predictions in enumerate(predictions):
        guess = np.argmax(predictions)
        actual = labels[i]
        confusion_matrix[actual][guess] += 1


    accuracies = np.zeros(shape=(nb_classes))
    for i in xrange(len(accuracies)):
        accuracies[i] = confusion_matrix[i][i] / float(np.sum(confusion_matrix[i]))
    return confusion_matrix, accuracies

def get_pred(sess, model, x):
    '''
    Returns predictions
    '''
    BATCH = get_batch_size(sess, model, x)
    results = []
    if BATCH > x.shape[0]:
        feed_dict = {model.x: x[:]}
        probabilities = tf.nn.softmax(model.model_logits)
        results = sess.run(probabilities, feed_dict=feed_dict).tolist()
    else:
        for i in xrange(int(x.shape[0] / BATCH)):
            feed_dict = {model.x: x[i * BATCH:(i + 1) * BATCH]}
            probabilities = tf.nn.softmax(model.model_logits)
            sub_results = sess.run(probabilities, feed_dict=feed_dict)
            # sub_results = sess.run(model.model_logits, feed_dict=feed_dict)
            results += sub_results.tolist()
        # finalize last sub-batch
        if x.shape[0] % BATCH != 0:
            feed_dict = {model.x: x[(i + 1) * BATCH:]}
            probabilities = tf.nn.softmax(model.model_logits)
            sub_results = sess.run(probabilities, feed_dict=feed_dict)
            results += sub_results.tolist()
    return results

def get_predictions(sess, model, x, symmetry):
    '''
    Returns predictions
    '''
    model.training = False
    results = []
    if not symmetry:
        results.append(get_pred(sess, model, x))
    else:
        li = []
        for rotations in xrange(4):
            for flips in xrange(4):
                data = augment_dataset(x, rotations, flips)
                li.append(get_pred(sess, model, data))
        for row in li:
            if row not in results:
                results.append(row)
    return list(np.mean(results, axis=0))


def run_session_to_get_accuracy(model, x, y):
    '''
    Get accuracy in batches to prevent resource exhausted error
    '''
    model.training = False

    BATCH = 200
    accuracy = 0

    for i in xrange(int(x.shape[0] / BATCH)):
        feed_dict = {model.x: x[i * BATCH:(i + 1) * BATCH], model.y: y[i * BATCH:(i + 1) * BATCH]}
        accuracy += model.accuracy.eval(feed_dict)
    test_accuracy = accuracy / int(x.shape[0] / BATCH)

    model.training = True
    return test_accuracy


def accuracy(results, y):
    '''
    Return accuracy.
    * results: predictions. list of lists. List within list is [value neg, value pos]
    * y: ground truth. list of lists. List within list is [prob neg, prob pos]
    '''
    total, correct = (0, 0)
    for i, result in enumerate(results):
        total += 1
        if np.argmax(result) == np.argmax(y[i]):
            correct += 1
    return correct / float(total)


def confusion_matrix(results, y):
    '''
    Get confusion matrix in the form of
    TN FP
    FN TP
    '''
    conf_matrix = np.zeros(shape=(y.shape[1], y.shape[1]))
    for i, result in enumerate(results):
        prediction = np.argmax(result)
        actual = np.argmax(y[i])
        conf_matrix[actual][prediction] += 1
    return conf_matrix


def get_metrics(conf):
    '''
    Get true positive, true negative, false positve and false negatives from confusion matrix
    Calculate accuracy, precision, and sensitivity based on these values.
    '''
    tp, tn = conf[1][1], conf[0][0]
    fp, fn = conf[0][1], conf[1][0]

    accuracy = (tp + tn) / float(sum(sum(conf)))
    precision = (tp) / float(tp + fp)
    sensitivity = (tp) / float(fn + tp)
    fp_rate = (fp) / float(tp)

    return accuracy, precision, sensitivity, fp_rate


def accuracy_manual(sess, model, x, y):
    model.training = False
    correct = 0
    total = 0

    for i in xrange(int(x.shape[0] / BATCH)):
        feed_dict = {model.x: x[i * BATCH:(i + 1) * BATCH]}
        results = sess.run(model.model_logits, feed_dict=feed_dict)
        for j, result in enumerate(results):
            total += 1
            if np.argmax(results[j]) == np.argmax(model.data.test.y[(i * BATCH) + j]):
                correct += 1
    return correct / float(total)


def get_probabilities(result, model_weight=1):
    result = np.array(result)
    differences = [abs(r[0] - r[1]) for r in result]
    max_diff, min_diff = max(differences), min(differences)
    probabilities = np.zeros(shape=(len(differences), 2))

    lowest_value = 1.0
    for i, elem in enumerate(result):
        elem_probability = (differences[i] - min_diff) / (max_diff - min_diff)
        elem_probability = elem_probability * model_weight

        if elem_probability == 0:
            elem_probability = 100  # hack
        if elem_probability < lowest_value:
            lowest_value = elem_probability

        probabilities[i, np.argmax(elem)] = elem_probability

    probabilities[probabilities == 100] = lowest_value - 0.01  # ensure lowest value
    return probabilities


def transform_probabilities(probabilities):
    sum_prob = probabilities[0]
    for i in xrange(1, len(probabilities)):
        sum_prob += probabilities[i]
    return get_probabilities(sum_prob)


def ensemble_submission(results, model_weights, data, log=None):
    # get probabilities
    probabilities = [get_probabilities(r, model_weights[i]) for i, r in enumerate(results)]
    probabilities = transform_probabilities(probabilities)

    conf_matrix = confusion_matrix(probabilities, data.y)
    a, p, s, fp_rate = get_metrics(conf_matrix)

    return probabilities, conf_matrix, a, p, s, fp_rate


def evaluate_ensemble(results, data, log=None):
    correct = 0
    models = len(results)
    for i in xrange(len(results[0])):
        preds = []
        for j in xrange(models):
            elem = results[j][i]
            preds.append(np.argmax(elem))
        pred = 1 if np.sum(preds) > (models / 2.) else 0
        if pred == np.argmax(data.y[i]):
            correct += 1
    accuracy = correct / float(len(results[0]))
    msg = 'Ensemble model: %.2f' % accuracy
    if log is not None:
        log.result(msg)
    else:
        print(msg)
