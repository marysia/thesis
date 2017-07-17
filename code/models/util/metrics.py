import numpy as np

def get_predictions(sess, logits, x, test_set_x, test_set_y):
    results = sess.run(logits, feed_dict={x: test_set_x})
    for i, result in enumerate(results):
        print np.argmax(result), np.argmax(test_set_y[i])


def confusion_matrix(sess, logits, x, test_set_x, test_set_y, training):
    '''
    Returns a confusion matrix with the actual labels on the row
    axis and the found prediction on the columns.
    '''
    conf_matrix = np.zeros(shape=[test_set_y.shape[1], test_set_y.shape[1]])

    results = sess.run(logits, feed_dict={x: test_set_x, training: False})
    for i, result in enumerate(results):
        prediction = np.argmax(result)
        actual = np.argmax(test_set_y[i])
        conf_matrix[actual][prediction] += 1
    return conf_matrix

def accuracy(sess, logits, x, test_set_x, test_set_y, training):
    correct = 0
    total = 0
    results = sess.run(logits, feed_dict={x: test_set_x, training: False})
    for i, result in enumerate(results):
        total += 1
        if np.argmax(result) == np.argmax(test_set_y[i]):
            correct += 1
    return correct/float(total)

def classes(conf_matrix):
    for i, row in enumerate(conf_matrix):
        print 'Class ', i,
        print ' correct: ', row[i] / float(sum(row)),
        row[i] = 0
        print ' best match: ', np.argmax(row)