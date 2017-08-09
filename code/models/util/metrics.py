import numpy as np

BATCH = 200

def get_predictions(sess, model, x):
    ''' Get list of lists based on trained model.
    Process in small batches to prevent resource exhausted error.'''
    model.training = False
    results = []
    for i in xrange(int(x.shape[0] / BATCH)):
        feed_dict = {model.x: x[i * BATCH:(i + 1) * BATCH]}
        sub_results = sess.run(model.model_logits, feed_dict=feed_dict)
        results += sub_results.tolist()
    return results

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
    ''' Get true positive, true negative, false positve and false negatives from confusion matrix
    Calculate accuracy, precision, and sensitivity based on these values. '''
    tp, tn = conf[1][1], conf[0][0]
    fp, fn = conf[0][1], conf[1][0]

    accuracy = (tp + tn) / float(sum(sum(conf)))
    precision = (tp) / float(tp + fp)
    sensitivity = (tp) / float(tn + tp)
    return accuracy, precision, sensitivity


def classes(conf_matrix):
    ''' old. '''
    for i, row in enumerate(conf_matrix):
        print 'Class ', i,
        print ' correct: ', row[i] / float(sum(row)),
        row[i] = 0
        print ' best match: ', np.argmax(row)

def get_accuracy(model, x, y):
    ''' Get accuracy in BATCHes to prevent resource exhausted error '''
    model.training = False

    BATCH = 200
    accuracy = 0

    for i in xrange(int(x.shape[0]/BATCH)):
        feed_dict = {model.x: x[i * BATCH:(i + 1) * BATCH], model.y: y[i * BATCH:(i + 1) * BATCH]}
        accuracy += model.accuracy.eval(feed_dict)
    test_accuracy = accuracy / int(x.shape[0] / BATCH)

    model.training = True
    return test_accuracy

def accuracy_manual(sess, model, x, y):
    model.training = False
    correct = 0
    total = 0

    for i in xrange(int(x.shape[0] / BATCH)):
        feed_dict = {model.x: x[i * BATCH:(i + 1) * BATCH]}
        results = sess.run(model.model_logits, feed_dict=feed_dict)
        for j, result in enumerate(results):
            total += 1
            if np.argmax(results[j]) == np.argmax(model.data.test.y[(i*BATCH)+j]):
                correct += 1
    return correct/float(total)

