import numpy as np

BATCH = 200

def get_predictions(sess, model, x):
    model.training = False
    results = []
    for i in xrange(int(x.shape[0] / BATCH)):
        feed_dict = {model.x: x[i * BATCH:(i + 1) * BATCH]}
        sub_results = sess.run(model.model_logits, feed_dict=feed_dict)
        results += sub_results.tolist()
    return results

def accuracy(results, y):
    total, correct = (0, 0)
    for i, result in enumerate(results):
        total += 1
        if np.argmax(result) == np.argmax(y[i]):
            correct += 1
    return correct / float(total)

def confusion_matrix(results, y):
    conf_matrix = np.zeros(shape=(y.shape[1], y.shape[1]))
    for i, result in enumerate(results):
        prediction = np.argmax(result)
        actual = np.argmax(y[i])
        conf_matrix[actual][prediction] += 1
    return conf_matrix

def get_metrics(conf):
    tp, tn = conf[1][1], conf[0][0]
    fp, fn = conf[0][1], conf[1][0]

    accuracy = (tp + tn) / float(sum(sum(conf)))
    precision = (tp) / float(tp + fp)
    recall = (tp) / float(tn + tp)
    return accuracy, precision, recall

def get_predictions2(sess, logits, x, test_set_x, test_set_y):
    results = sess.run(logits, feed_dict={x: test_set_x})
    for i, result in enumerate(results):
        print np.argmax(result), np.argmax(test_set_y[i])


def confusion_matrix2(sess, logits, x, test_set_x, test_set_y):
    '''
    Returns a confusion matrix with the actual labels on the row
    axis and the found prediction on the columns.
    '''
    conf_matrix = np.zeros(shape=[test_set_y.shape[1], test_set_y.shape[1]])

    results = sess.run(logits, feed_dict={x: test_set_x})
    for i, result in enumerate(results):
        prediction = np.argmax(result)
        actual = np.argmax(test_set_y[i])
        conf_matrix[actual][prediction] += 1
    return conf_matrix

def classes(conf_matrix):
    for i, row in enumerate(conf_matrix):
        print 'Class ', i,
        print ' correct: ', row[i] / float(sum(row)),
        row[i] = 0
        print ' best match: ', np.argmax(row)

def accuracy3(model, x, y):
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

