import tensorflow as tf
import numpy as np
import util.metrics
from util.helpers import total_parameters, progress
from util.augmentation import rotate_transform_batch

import sys
import time


class BaseModel:
    def __init__(self, model_name, data, ender, log, x_val=None, y_val=None,
                 epochs=5, batch_size=128, learning_rate=0.001, verbose=True):
        self.model_name = model_name
        self.verbose = verbose
        self.ender = ender
        self.log = log

        self.transformations = ['rotation']

        self.training = True
        self.optimizer = None
        self.learning_rate = learning_rate

        self.model_logits = None

        self.data = data

        self.epochs = epochs
        self.batch_size = batch_size
        self.steps = self.data.train.x.shape[0] / self.batch_size

    def build_graph(self):
        ''' To be implemented in the subclass. Sets self.model_logits.'''
        raise NotImplementedError

    def build_model(self):
        '''
        Builds the model by setting placeholders, building the graph (see subclass),
        and defining cross entropy, optimizer (adam), correct prediction and accuracy calculation.
        '''
        with tf.name_scope('Placeholders'):
            self.x = tf.placeholder(tf.float32, shape=[None] + list(self.data.train.x.shape[1:]))
            self.y = tf.placeholder(tf.float32, shape=[None, self.data.nb_classes])

        self.build_graph()

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.model_logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.model_logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        self.log.info('Model: Initizalization done...')

    def train(self, mode='epochs', mode_param=None, save_step=5):
        ''' Trains the model.
        Builds the model, initializes the global variables, and executes the train step
        for as long as the mode specifies.

        Modes:
            - epochs: runs through the entire training set in batches of self.batch_size for self.epochs epochs
            - converge: runs through the entire training set in batches of self.batch_size for as long as the
            there is still a relatively large difference between results on the validation set (defined as:
            largest difference from mean of the last ten results is bigger than 0.01)

        '''
        # initialize start time and set self.training to true for dropout.
        start_time = time.time()
        self.training = True

        # check for modes.
        if mode not in ['epochs', 'converge', 'time']:
            self.log.info('Mode not recognised. Please use epochs, converge or time.')
            raise Exception
        if mode == 'converge' and self.data.val.scope == 'val-empty':
            self.log.info('To use the converge mode, a validation set must be provided.')
            raise Exception

        # initialize variables
        init = tf.global_variables_initializer()

        # begin session
        with tf.Session() as sess:
            # initialize variables
            sess.run(init)

            if mode == 'epochs':
                self.epochs = mode_param if mode_param is not None else self.epochs
                # run through entire training self.epochs times.
                i = 0
                while not self.ender.terminate and i < self.epochs:
                    i += 1
                    # for epoch in range(self.epochs):
                    # perform training step
                    prefix = '\rEpoch %d of %d: ' % (i, self.epochs)
                    result_train, result_val = self._train_step(prefix)

                    # print if necessary
                    #if self.verbose:
                    #    self.log.info('\nEpoch %d of %d: train accuracy: %g, validation accuracy: %g' % (
                    #    i, self.epochs, result_train, result_val))

                    # save to log if necessary
                    if i % save_step == 0:
                        self.training = False
                        accuracy = self.get_accuracy(self.data.test.x, self.data.test.y)
                        self.log.info('Model: %s, step %s, test accuracy: %g' % (self.model_name, i, accuracy))
                        self.training = True

                if self.ender.terminate:
                    self.log.info('Program was terminated after %d epochs. Exiting gracefully.' % i)


            if mode == 'converge':
                max_difference = mode_param if mode_param is not None else 0.01
                # initialize variables
                i = 0
                val_results = []
                # stop if the highest difference from mean (of last ten results on val set)  is smaller than 0.01
                while i < 10 or np.max(np.abs(val_results[-10:] - np.mean(val_results[-10:]))) > max_difference:
                    # perform training step
                    prefix = '\rIteration %d: ' % (i + 1)
                    result_train, result_val = self._train_step(prefix)

                    # update variables
                    val_results.append(result_val)
                    i += 1

                    # print if necessary
                    if self.verbose:
                        self.log.info(
                            '\nIteration %d: train accuracy: %g, validation accuracy: %g, max deviation: %.2f') \
                        % (i, result_train, result_val, np.max(np.abs(val_results[-10:] - np.mean(val_results[-10:]))))

            if mode == 'time':
                # default: 5 minutes.
                mode_param = mode_param if mode_param is not None else 5
                base_time = time.time()
                last_save = base_time
                end_time = base_time + (mode_param * 60)
                i = 0
                while time.time() < end_time and not self.ender.terminate:
                    prefix = '\rIteration %d: ' % (i + 1)
                    result_train, result_val = self._train_step(prefix)
                    #if self.verbose:
                    #    self.log.info('\nEpoch %d of %d: train accuracy: %g, validation accuracy: %g' % (
                    #    i, self.epochs, result_train, result_val))
                    i += 1

                    if ((time.time() - last_save) / 60) > save_step:
                        last_save = time.time()
                        self.training = False
                        accuracy = self.get_accuracy(self.data.test.x, self.data.test.y)
                        self.log.result('Model: %s, step %s, test accuracy: %g' % (self.model_name, i, accuracy))
                        self.training = True

                if self.ender.terminate:
                    exiting_time = (time.time() - base_time) / 60
                    self.log.info('Program was terminated after %d minutes (%d iterations). Exiting gracefully.' % (
                    exiting_time, i))
                else:
                    self.log.info('Performed %d iterations in %d minutes.' % (i, mode_param))

            if i > 0:
                self.evaluate(sess)
            elapsed_time = (time.time() - start_time) / 60
            self.log.info('Training took approximately %d minutes.' % elapsed_time)

    def _train_step(self, prefix):
        for step in range(self.steps):
            batch = self.data.train.get_next_batch(step, self.batch_size)

            if 'rotation' in self.transformations:
                batch.x = rotate_transform_batch(batch.x, rotation = 2 * np.pi)

            self.optimizer.run(feed_dict={self.x: batch.x, self.y: batch.y})

            if self.verbose:
                progress(prefix, step, self.steps)
        # return 0, 0
        result_train, result_val = (0, 0)
        #if self.verbose:
        #    result_train = self.get_accuracy(self.data.train.x, self.data.train.y)
        #    result_val = self.get_accuracy(self.data.val.x,
        #                                   self.data.val.y) if not 'empty' in self.data.val.scope else 0

        return result_train, result_val

    def get_accuracy(self, x, y):
        self.training = False
        try:
            feed_dict = {self.x: x, self.y: y}
            test_accuracy = self.accuracy.eval(feed_dict)
        except:
            batch = 200
            accuracy = 0
            # for i in xrange(x.shape[0] / batch):
            for i in xrange(int(x.shape[0] / batch)):
                feed_dict = {self.x: x[i * batch:(i + 1) * batch], self.y: y[i * batch:(i + 1) * batch]}
                accuracy += self.accuracy.eval(feed_dict)
            test_accuracy = accuracy / int(x.shape[0] / batch)
        self.training = True
        return test_accuracy

    def evaluate(self, sess):
        self.training = False
        if self.verbose:
            pass
            # print('Test accuracy: %g' % self.accuracy.eval(feed_dict={self.x: self.data.test.x, self.y: self.data.test.y}))
            # print util.metrics.accuracy(sess, self.model_logits, self.x, self.data.test.x, self.data.test.y, self.training)
            # conf_matrix = util.metrics.confusion_matrix(sess, self.model_logits, self.x, self.data.test.x, self.data.test.y)
            # print conf_matrix
            # print util.metrics.classes(conf_matrix)
            # print('Done.')
            # conf_matrix = util.metrics.confusion_matrix(sess, self.model_logits, self.x, self.data.test.x, self.data.test.y)
            # util.metrics.classes(conf_matrix)

        accuracy = self.get_accuracy(self.data.test.x, self.data.test.y)
        #conf_matrix = util.metrics.confusion_matrix(sess, self.model_logits, self.x, self.data.test.x, self.data.test.y)
        self.log.result('Model: %s, test accuracy: %g' % (self.model_name, accuracy))
        self.log.result('Model: %s, confusion matrix: ' % (self.model_name))
        #self.log.result(conf_matrix[0])
        #self.log.result(conf_matrix[1])
