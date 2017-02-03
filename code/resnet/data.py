import numpy as np
from keras.utils import np_utils


# --- helper functions --- #
def shuffle_in_unison(a, b):
    ''' Shuffles images and labels in unison. '''
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b

def center_crop(patches, size):
    ''' Crops patches to create 7x72x72 images. '''
    shape = patches.shape
    crop = (shape[2] - 72)/2
    patches = patches[0:size, :, crop:shape[2]-crop, crop:shape[2]-crop]
    return patches

def convert_data(pos, neg, size):
    ''' Crops, ceates labels and normalizes images.'''
    pos = center_crop(pos, size)
    pos_labels = np.ones(len(pos))
    neg = center_crop(neg, size)
    neg_labels = np.zeros(len(neg))

    X = np.concatenate([pos, neg])
    Y = np.append(pos_labels, neg_labels)

    # Data alterations
    X, Y = shuffle_in_unison(X, Y)  # shuffle indices

    X = X.reshape([-1, 7, 72, 72, 1])

    Y = np_utils.to_categorical(Y)

    X = (X - 127.5) / 127.5
    X = X.astype('float32')
    return X, Y

def normalize(X, Y):
    # Data alterations
    X, Y = shuffle_in_unison(X, Y)  # shuffle indices
    X = X.reshape([-1, 7, 72, 72, 1])
    Y = np_utils.to_categorical(Y)

    X = (X - 127.5) / 127.5
    X = X.astype('float32')
    return X, Y

# --- data conversion functions --- #
def lidc():
    ''' Normalize, reshape and save LIDC positives and negatives with labels.
    Create train and test set.'''
    datadir = '/home/marysia/thesis/data/'
    positives_data = np.load(datadir + 'lidc_positives.npz')['lidc_positives']
    positives_labels = np.ones(len(positives_data))

    negatives_data = np.load(datadir + 'lidc_negatives.npz')['patch']
    negatives_labels = np.zeros(len(negatives_data))
    #
    # X = np.concatenate([positives_data, negatives_data])
    # Y = np.append(positives_labels, negatives_labels)

    xtrain = np.concatenate([positives_data[0:1000], negatives_data[0:1000]])
    xtest = np.concatenate([positives_data[1000:], negatives_data[1000:]])

    ytrain = np.append(positives_labels[:1000], negatives_labels[:1000])
    ytest = np.append(positives_labels[1000:], negatives_labels[1000:])

    xtrain, ytrain = normalize(xtrain, ytrain)
    xtest, ytest = normalize(xtest, ytest)

    np.savez('/home/marysia/thesis/data/lidc.npz', xtrain=xtrain, xtest=xtest, ytrain=ytrain, ytest=ytest)



def candidates():
    ''' Normalize, reshape and save second phase candidates positives and negatives with labels.
    Create train and test set.'''
    datadir = '/home/ubuntu/data/gerben/patches/'
    data = np.load(datadir+'candidate-patches-14-56k-no-clahe.npz')
    print('Loaded.')
    pos = data['pos_examples']
    neg = data['neg_examples']
    xtrain, ytrain = convert_data(pos, neg, 8924)
    print('Converted train set.')

    pos = data['pos_examples_test']
    neg = data['neg_examples_test']
    xtest, ytest = convert_data(pos, neg, 2230)
    print('Converted test set.')

    np.savez('/home/marysia/thesis/data/second_phase.npz', xtrain=xtrain, xtest=xtest, ytrain=ytrain, ytest=ytest)
    print('Saved.')


# --- load functions --- #
def load_lidc(ordering='tf'):
    data = np.load('/home/marysia/thesis/data/lidc.npz')
    xtrain = data['xtrain']
    xtest = data['xtest']
    ytrain = data['ytrain']
    ytest = data['ytest']


    if ordering == 'th':
        xtrain = xtrain.reshape([-1, 1, 7, 72, 72])
        xtest = xtest.reshape([-1, 1, 7, 72, 72])

    return xtrain, ytrain, xtest, ytest


def load_candidates(ordering='tf'):
    data = np.load('/home/marysia/thesis/data/candidates.npz')
    xtrain = data['xtrain']
    xtest = data['xtest']
    ytrain = data['ytrain']
    ytest = data['ytest']

    if ordering == 'th':
        xtrain = xtrain.reshape([-1, 1, 7, 72, 72])
        xtest = xtest.reshape([-1, 1, 7, 72, 72])

    return xtrain, ytrain, xtest, ytest
