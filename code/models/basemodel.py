import time

import tensorflow as tf
import numpy as np

from util.augmentation import augment_batch
from util.metrics import get_predictions
from util.helpers import progress


class BaseModel:
    def __init__(self, name, group, data):
        self.name = name
        self.group = group
        self.data = data

        with tf.name_scope('Placeholders'):
            self.x = tf.placeholder(tf.float32, shape=[None] + self.data.shape + [1])
            self.y = tf.placeholder(tf.float32, shape=[None, self.data.nb_classes])

        self.training = True
        self.model_logits = self.build_graph()

    def build_graph(self):
        ''' To be implemented in subclass. Returns logits. '''
        raise NotImplementedError

    def set_variables(self, args, ender, log):
        ''' Set all necessary variables for training. '''
        self.ender = ender
        self.log = log

        self.submission = args.submission
        self.verbose = args.verbose
        self.symmetry = args.symmetry
        self.discard = args.discard
        self.mode_param = args.mode_param
        self.save_fraction = args.save_fraction

        self.augmentation = args.augment

        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.steps = (self.data.train.samples / self.batch_size)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.model_logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)

        self.meta = {}
        self.losses = []
        self.validation_losses = []
        self.progress = 0

        self.training = True    # for dropout

    def train(self, args, ender, log):
        ''' Function to call to train the model. '''
        self.set_variables(args, ender, log)
        start_time = time.time()

        init = tf.global_variables_initializer()

        with tf.Session() as self.sess:
            self.sess.run(init)

            if args.mode == 'epochs':
                iteration = self.train_epochs()
            elif args.mode == 'time':
                iteration = self.train_time()
            elif args.mode == 'batches':
                iteration = self.train_batches()

            self.training = False
            if iteration > 0:
                elapsed_time = (time.time() - start_time) / 60
                self.log.info('\nTraining took approximately %d minutes.' % elapsed_time, time=True)
                #self.evaluate()

    def train_epochs(self):
        ''' Train for _epochs_ number of epochs. '''
        iteration = 0
        while not self.ender.terminate and iteration < self.mode_param:
            iteration += 1
            prefix = '\rEpoch %d of %d: ' % (iteration, self.mode_param)
            self.train_step(prefix)
            self.progress += 1
            self.validation_loss()

        if self.ender.terminate:
            self.log.info('Program was terminated after %d epochs. Exiting gracefully.' % iteration, time=True)
        return iteration

    def train_time(self):
        ''' Train for _minutes_ number of minutes. '''
        iteration = 0

        base_time = time.time()
        previous_time = base_time
        end_time = base_time + (self.mode_param * 60)

        while not self.ender.terminate and time.time() < end_time:
            iteration += 1
            prefix = '\rIteration %d: ' % iteration
            self.train_step(prefix)
            self.progress += (time.time() - previous_time) / 60
            previous_time = time.time()

        if self.ender.terminate:
            exiting_time = (time.time() - base_time) / 60
            self.log.info('Program was terminated after %d minutes (%d iterations). Exiting gracefully.' % (
            exiting_time, iteration), time=True)
        else:
            self.log.info('Performed %d iterations in %d minutes.' % (iteration, self.mode_param), time=True)

        return iteration

    def train_batches(self):
        ''' Train for a specific amount of batches. '''
        self.mode_param *= 100
        cost = 0
        cost_batch = 10

        nr_epochs = (self.mode_param * self.batch_size) / float(self.data.train.samples)
        print('Training for %.2f epochs.' % nr_epochs )

        while not self.ender.terminate and self.progress < self.mode_param:

            step = self.progress % self.steps
            batch_x, batch_y = self.data.train.get_next_batch(step, self.batch_size)
            batch_x = augment_batch(batch_x, self.augmentation, self.data.shape)

            # run optimizer
            feed_dict = {self.x: batch_x, self.y: batch_y}
            _, c = self.sess.run([self.optimizer, self.cross_entropy], feed_dict=feed_dict)
            cost += c

            self.progress += 1

            if self.progress % cost_batch == 0:
                cost /= cost_batch
                self.losses.append(cost)
                cost = 0

                self.validation_loss()

            if self.verbose and self.progress % 10 == 0:
                progress('\r%d of %d' % (self.progress, self.mode_param), self.progress, self.mode_param)


        iteration = self.progress / float(self.steps)
        print(self.validation_losses)
        return iteration


    def train_step(self, prefix):
        ''' Perform training step. Retrieve batch, apply augmentations and run optimizer. '''
        avg_cost_train = 0
        for step in xrange(self.steps):
            # retrieve batch
            batch_x, batch_y = self.data.train.get_next_batch(step, self.batch_size)
            batch_x = augment_batch(batch_x, self.augmentation, self.data.shape)

            # run optimizer
            feed_dict = {self.x: batch_x, self.y: batch_y}

            _, cost_train = self.sess.run([self.optimizer, self.cross_entropy], feed_dict=feed_dict)
            avg_cost_train += cost_train

            # report progress
            if self.verbose:
                progress(prefix, step, self.steps)

        avg_cost_train /= self.steps
        self.losses.append(avg_cost_train)

    def validation_loss(self):
        ''' Calculates the loss on the validation set.
        Additionally, after a fraction of training has been done, keeps track
        of the lowest loss and evaluates if necessary. '''
        self.training = False
        if 'empty' not in self.data.val.scope:
            cost = 0
            batch_size = 100 if self.data.val.samples > 100 else self.data.val.samples
            steps = self.data.val.samples / batch_size
            for i in xrange(steps):
                x, y = self.data.val.get_next_batch(i, batch_size)
                cost += self.sess.run([self.cross_entropy], feed_dict={self.x: x, self.y: y})[0]
            cost /= steps
            self.validation_losses.append(cost)

            if (self.progress / float(self.mode_param)) > self.save_fraction:
                if (self.progress / float(self.mode_param) == 1) and ('permutation-test-set' not in self.meta.keys()):
                    self.log.info('\nReached end of train cycle; evaluating validation and test set.')
                    self.evaluate()
                elif cost <= min(self.validation_losses):
                    self.log.info('\nLowest validation loss: %.4f at %.2f per cent.' % (cost, (self.progress / float(self.mode_param))*100))
                    self.evaluate()
        elif (self.progress / float(self.mode_param) == 1):
            self.evaluate()

        self.training = True


    def set_meta(self, iteration, parameters):
        ''' Sets metadata to be saved in pickle later on. '''
        if not self.discard:
            # model info
            self.meta['name'] = self.name
            self.meta['group'] = self.group

            # iteration info
            self.meta['iteration'] = iteration
            self.meta['parameters'] = parameters
            self.meta['log-identifier'] = self.log.runid

            # train info
            self.meta['augmentations'] = self.augmentation
            self.meta['batch-size'] = self.batch_size
            self.meta['learning-rate'] = self.learning_rate

            # dataset info
            self.meta['train-dataset'] = self.data.train_dataset
            self.meta['val-dataset'] = self.data.val_dataset
            self.meta['test-dataset'] = self.data.test_dataset

            self.meta['training-set-samples'] = self.data.train.samples
            self.meta['data-shape'] = self.data.shape
            self.meta['val-set-samples'] = self.data.val.samples
            self.meta['test-set-samples'] = self.data.test.samples

            self.meta['train-loss'] = self.losses
            self.meta['val-loss'] = self.validation_losses

        
    def evaluate(self):
        ''' Saves validation and test set results to meta variable
            Saved per set:
                * Permutation of data shuffle to retrieve original series uid
                * Labels (not one-hot encoded)
                * Predictions
                * Predictions averaged over symmetry
        '''
        if not self.discard:
            if self.data.val.scope != 'val-empty':
                self.meta['permutation-val-set'] = self.data.val.id
                self.meta['labels-val-set'] = np.argmax(self.data.val.y, axis=1)
                self.meta['val-predictions'] = get_predictions(self.sess, self, self.data.val.x, False)
                if self.symmetry:
                    self.meta['val-symmetry-predictions'] = get_predictions(self.sess, self, self.data.val.x, True)

            if self.data.test.scope != 'test-empty':
                self.meta['permutation-test-set'] = self.data.test.id
                self.meta['labels-test-set'] = np.argmax(self.data.test.y, axis=1)
                self.meta['test-predictions'] = get_predictions(self.sess, self, self.data.test.x, False)
                if self.symmetry:
                    self.meta['test-symmetry-predictions'] = get_predictions(self.sess, self, self.data.test.x, True)


