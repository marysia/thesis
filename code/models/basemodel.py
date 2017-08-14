import tensorflow as tf
import numpy as np
import util.metrics
from util.helpers import total_parameters, progress
from util.augmentation import rotate_transform_batch2d, rotate_transform_batch3d

import sys
import time


class BaseModel:
    def __init__(self, model_name, data, ender, log, x_val=None, y_val=None,
                 epochs=5, batch_size=128, learning_rate=0.001, augmentation="", submission=False, verbose=True):
        self.model_name = model_name
        self.submission = submission
        self.verbose = verbose
        self.ender = ender
        self.log = log

        self.transformations = augmentation

        self.training = True
        self.optimizer = None
        self.learning_rate = learning_rate

        self.model_logits = None

        self.data = data

        self.epochs = epochs
        self.batch_size = batch_size
        steps = self.data.train.x.shape[0] / self.batch_size
        self.steps = steps * 2 if not self.data.balanced else steps
        #self.steps = steps * self.data.train.fraction if not self.data.balanced else steps

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
            - time: runs through the entire training set in batches of batch size for n minutes.

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
                i = self.train_epochs(sess, save_step)

            elif mode == 'time':
                i = self.train_time(sess, mode_param, save_step)

            elif mode == 'converge':
                i = self.train_converge()

            else:
                pass

            if i > 0:
                self.evaluate(sess)
            elapsed_time = (time.time() - start_time) / 60
            self.log.info('Training took approximately %d minutes.' % elapsed_time)

    # --- train modes --- #
    def train_epochs(self, sess, save_step):
        '''
        As long as no termination signal has been send, loop through the dataset self.epochs times
        and perform a train step. Every save_step epochs, calculate accuracy and log.
        '''
        i = 0
        while not self.ender.terminate and i < self.epochs:
            i += 1
            prefix = '\rEpoch %d of %d: ' % (i, self.epochs)
            self._train_step(prefix)

            if i % save_step == 0:
                if self.verbose:
                    print(' ')
                prefix = 'Model: %s, step %s' % (self.model_name, i)
                self.progress_metrics(sess, prefix)

        if self.ender.terminate:
            self.log.info('Program was terminated after %d epochs. Exiting gracefully.' % i)
        return i

    def train_time(self, sess, mode_param, save_step):
        '''
        Train for mode_param minutes, and save every save_step minutes.
        '''
        # initialize
        mode_param = mode_param if mode_param is not None else 5
        i = 0

        # set begin times
        base_time = time.time()   # start time
        last_save = base_time   # initialize last save to starting time
        end_time = base_time + (mode_param * 60)   # time training should end

        # while not passed training end time and no termination signal
        while time.time() < end_time and not self.ender.terminate:
            i += 1
            prefix = '\rIteration %d: ' % (i)

            self._train_step(prefix)

            # if time passed in minutes since last save moment is bigger than save step: save.
            if ((time.time() - last_save) / 60) > save_step:
                if self.verbose:
                    print(' ')
                last_save = time.time()   # set last save to current time
                prefix = 'Model: %s, step %s' % (self.model_name, i)
                self.progress_metrics(sess, prefix)

        if self.ender.terminate:
            exiting_time = (time.time() - base_time) / 60
            self.log.info('Program was terminated after %d minutes (%d iterations). Exiting gracefully.' % (
                exiting_time, i))
        else:
            self.log.info('Performed %d iterations in %d minutes.' % (i, mode_param))
        return i

    def train_converge(self, mode_param):
        '''
        Currently unused.
        '''
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

        return i

    def _train_step(self, prefix):
        ''' Training step. '''
        for step in range(self.steps):
            batch = self.data.train.get_next_batch(step, self.batch_size)

            if self.transformations == "rotation2d":
                x = rotate_transform_batch2d(batch.x, rotation = 2 * np.pi)
            elif self.transformations == "rotation3d":
                x = rotate_transform_batch3d(batch.x)
            else:
                x = batch.x


            self.optimizer.run(feed_dict={self.x: x, self.y: batch.y})

            if self.verbose:
                progress(prefix, step, self.steps)

    def progress_metrics(self, sess, prefix):
        results = util.metrics.get_predictions(sess, self, self.data.test.x)
        confusion_matrix = util.metrics.confusion_matrix(results, self.data.test.y)
        a, p, s, fp_rate = util.metrics.get_metrics(confusion_matrix)

        self.log.result('%s, accuracy: %.2f, sensitivity: %.2f, fp_rate: %.2f' % (prefix, a, s, fp_rate))

    def evaluate(self, sess):
        ''' Get evaluation metrics through util.metrics and log.'''

        results = util.metrics.get_predictions(sess, self, self.data.test.x)
        confusion_matrix = util.metrics.confusion_matrix(results, self.data.test.y)
        a, p, s, fp_rate = util.metrics.get_metrics(confusion_matrix)
        
        self.log.result('Acc: %.2f, Prec: %.2f, Sensitivity: %.2f, FP-rate: %.2f' % (a, p, s, fp_rate))
        self.log.result(util.helpers.pretty_print_confusion_matrix(confusion_matrix))

        if self.submission:
            util.helpers.create_submission(self.model_name, self.log,  self.data, results)

