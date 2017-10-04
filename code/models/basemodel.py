import os
import time

import numpy as np
import tensorflow as tf

import util.metrics
from util.augmentation import augment_batch
from util.helpers import progress, error_plot


class BaseModel:
    def __init__(self, name, group, data):
        self.model_name = name
        self.group = group
        self.data = data

        self.evaluation = {'weight': 0, 'results': [], 'mse': 0}

        with tf.name_scope('Placeholders'):
            self.x = tf.placeholder(tf.float32, shape=[None] + self.data.shape + [1])
            self.y = tf.placeholder(tf.float32, shape=[None, self.data.nb_classes])

        self.model_logits = self.build_graph()

    def build_graph(self):
        ''' To be implemented in the subclass. Sets self.model_logits.'''
        raise NotImplementedError

    def get_graph(self):
        return self.model_logits

    def set_variables(self, args, augmentation, ender, log):
        self.results = None
        self.ender = ender
        self.log = log

        self.submission = args.submission
        self.verbose = args.verbose
        self.symmetry = args.symmetry

        self.transformations = augmentation

        self.training = True

        self.batch_size = 32
        self.epochs = 5
        self.learning_rate = 0.0001
        self.lr_decay = 1.0

        self.model_weight = 0

        if self.data.train.balanced:
            self.steps = (self.data.train.samples / self.batch_size) + 1
        else:
            self.steps = len(self.data.train.pos_idx) / (self.batch_size / 2)

        self.softmax = tf.nn.softmax(self.model_logits)
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.model_logits))

        # self.cross_entropy = tf.losses.softmax_cross_entropy(self.y, self.model_logits)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)

    def train(self, args, augmentation, ender, log):
        ''' Trains the model.
        Executes the train step
        for as long as the mode specifies.

        Modes:
            - epochs: runs through the entire training set in batches of self.batch_size for self.epochs epochs
            - converge: runs through the entire training set in batches of self.batch_size for as long as the
            there is still a relatively large difference between results on the validation set (defined as:
            largest difference from mean of the last ten results is bigger than 0.01)
            - time: runs through the entire training set in batches of batch size for n minutes.

        '''
        self.set_variables(args, augmentation, ender, log)
        start_time = time.time()

        # initialize variables
        init = tf.global_variables_initializer()

        # begin session
        with tf.Session() as sess:
            # initialize variables
            sess.run(init)

            # saver
            saver = tf.train.Saver()

            if args.mode == 'epochs':
                self.epochs = args.mode_param if args.mode_param is not None else self.epochs
                i = self.train_epochs(sess, args.reinforce, args.save_step)

            elif args.mode == 'time':
                i = self.train_time(sess, args.mode_param, args.reinforce, args.save_step)

            elif args.mode == 'converge':
                i = self.train_converge(sess, args.mode_param)

            else:
                pass

            if i > 0:
                self.evaluate(sess)
            elapsed_time = (time.time() - start_time) / 60
            self.log.info('Training took approximately %d minutes.' % elapsed_time)

            if args.save_model:
                base_path = '/home/marysia/thesis/results/models/%s_%s/model'
                if args.ensemble:
                    model_path = os.path.join(base_path, 'ensemble-%s/model' % self.log.runid)
                else:
                    model_path = os.path.join(base_path, '%s_%s/model' % (self.model_name, self.log.runid))
                saver.save(sess, model_path)

    def load_model(self, modelpath):
        # TODO: to be implemented
        pass

    # --- train modes --- #
    def train_epochs(self, sess, reinforce, save_step):
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

                if reinforce:
                    self._reinforce(sess)

        if self.ender.terminate:
            self.log.info('Program was terminated after %d epochs. Exiting gracefully.' % i)
        return i

    def train_time(self, sess, mode_param, reinforce, save_step):
        '''
        Train for mode_param minutes, and save every save_step minutes.
        '''
        # initialize
        mode_param = mode_param if mode_param is not None else 5
        i = 0

        # set begin times
        base_time = time.time()  # start time
        last_save = base_time  # initialize last save to starting time
        end_time = base_time + (mode_param * 60)  # time training should end

        # while not passed training end time and no termination signal
        while time.time() < end_time and not self.ender.terminate:
            i += 1
            prefix = '\rIteration %d: ' % (i)

            self._train_step(prefix)

            # if time passed in minutes since last save moment is bigger than save step: save.
            if ((time.time() - last_save) / 60) > save_step:
                if self.verbose:
                    print(' ')
                last_save = time.time()  # set last save to current time
                prefix = 'Model: %s, step %s' % (self.model_name, i)
                self.progress_metrics(sess, prefix)

            if reinforce and (i % save_step) == 0:
                self._reinforce(sess)

        # termination signal was received
        if self.ender.terminate:
            exiting_time = (time.time() - base_time) / 60
            self.log.info('Program was terminated after %d minutes (%d iterations). Exiting gracefully.' % (
                exiting_time, i))
        else:
            self.log.info('Performed %d iterations in %d minutes.' % (i, mode_param))

        return i

    def train_converge(self, sess, mode_param):
        '''
        Train until the error on the validation set is more or less the same as the error on the
        training set.
        '''
        errors = []
        i = 0
        #while abs(current_mse - previous_mse) > 0.001 and not self.ender.terminate:
        while not self.ender.terminate:
            # Check every _mode_param_ steps.
            for _ in xrange(mode_param):
                prefix = '\rIteration %d: ' % (i + 1)
                self._train_step(prefix)
                i += 1

            validation_mse = util.metrics.mse(sess, self, self.data.val.x, self.data.val.y)
            training_mse = util.metrics.mse(sess, self, self.data.train.x, self.data.train.y)
            errors.append((training_mse, validation_mse))
            self.log.info('\n%sTraining error: %.3f \t Validation error: %.3f' % (prefix, training_mse, validation_mse))
            if not validation_mse < training_mse:
                break

        util.helpers.error_plot(errors, mode_param)


    def _reinforce(self, sess):
        ''' Run incorrect batches an extra time '''
        # get results for training set
        results = util.metrics.run_session_to_get_predictions(sess, self, self.data.train.x)
        # get indices of all results that aren't correct
        idx = [i for i, pred in enumerate(results) if np.argmax(pred) != np.argmax(self.data.train.y[i])]

        # shuffle
        p = np.random.permutation(len(idx))
        idx = np.array(idx)[p]

        # divide in batches
        nr_batches = int(len(idx) / self.batch_size)
        self.log.info('Reinforcing incorrect training samples: %d batches.' % nr_batches)
        for i in range(nr_batches):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            batch_x = self.data.train.x[start:end]
            batch_y = self.data.train.y[start:end]

            self.optimizer.run(feed_dict={self.x: batch_x, self.y: batch_y})

    def _train_step(self, prefix):
        ''' Training step. Retrieve batch, apply augmentation and run optimizer.'''
        for step in range(self.steps):
            batch_x, batch_y = self.data.train.get_next_batch(step, self.batch_size)

            batch_x = augment_batch(batch_x, self.transformations, self.data.shape)

            self.optimizer.run(feed_dict={self.x: batch_x, self.y: batch_y})

            if self.verbose:
                progress(prefix, step, self.steps)
        # self.learning_rate = self.learning_rate * self.lr_decay

    def progress_metrics(self, sess, prefix):
        ''' Log progress during training, preferably on validation set but otherwise test set.'''
        x, y = (self.data.val.x, self.data.val.y) if not 'empty' in self.data.val.scope else (
            self.data.test.x, self.data.test.y)

        # predictions, confusion matrix, accuracy, precision, sensitivity and false positive rate.
        results = util.metrics.run_session_to_get_predictions(sess, self, x, False)
        confusion_matrix = util.metrics.confusion_matrix(results, y)
        a, p, s, fp_rate = util.metrics.get_metrics(confusion_matrix)

        self.evaluation['weight'] = a   # weight of model = accuracy on validation set
        self.log.result('%s, accuracy: %.2f, sensitivity: %.2f, fp_rate: %.2f' % (prefix, a, s, fp_rate))

    def evaluate(self, sess):
        ''' Get evaluation metrics through util.metrics and log at the end of the training run.'''
        # get results on validation set.
        self.progress_metrics(sess, '\nValidation set results: ')

        # predictions, confusion matrix, accuracy, precision, sensitivity and false positive rate.
        results = util.metrics.run_session_to_get_predictions(sess, self, self.data.test.x, self.symmetry)
        confusion_matrix = util.metrics.confusion_matrix(results, self.data.test.y)
        a, p, s, fp_rate = util.metrics.get_metrics(confusion_matrix)

        self.log.result('Acc: %.2f, Prec: %.2f, Sensitivity: %.2f, FP-rate: %.2f' % (a, p, s, fp_rate))
        self.log.result(util.helpers.pretty_print_confusion_matrix(confusion_matrix))

        #if self.submission:
        #    util.helpers.create_submission(self.model_name, self.log, self.data.test, results)

        # set results to evaluation parameter (dictionary)
        self.evaluation['results'] = results
        #self.evaluation['mse'] = util.metrics.mse(sess, self, self.data.test.x, self.data.test.y)
        self.evaluation['sensitivity'] = s
        self.evaluation['fp-rate'] = fp_rate
        self.evaluation['accuracy'] = a
