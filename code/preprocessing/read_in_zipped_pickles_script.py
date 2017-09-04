# Candidates generated by the localization model are saved in a patches-?.pkl.zip format. These zipped pickle files were
# written using protocol 4, which is not supported by Python 2.7, and cannot be loaded. Therefore, this python3 script
# reads in all zipped pickle files and converts them to a format that can be loaded using Python 2.7.
# Note: preferably run this script from command line with sudo, as a permission denied error otherwise occurs for the
# /home/marysia/data folder on Azure.
import glob

import numpy as np

from code.preprocessing.zippedpickles import load


def extract_lidc_localization_samples(patch_folder):
    print(patch_folder)
    patches = glob.glob(patch_folder + '*.pkl.zip')

    print('Extracting samples from %d patches.' % (len(patches)))

    for label in ['positive', 'negative']:
        scans = []
        metadata = []
        data = []
        for i, patch_path in enumerate(patches):
            try:
                print('%d/%d: %s' % (i + 1, len(patches), patch_path))
                # loading only loads the keys, therefore takes little time
                unzipped = load(patch_path)

                # train:
                scans += unzipped['train-scans']
                if label == 'positive':
                    metadata += unzipped['train-%s-metadata' % label]

                a = unzipped['train-%s-inputs' % label]
                print('train', label, a.shape)
                data.append(unzipped['train-%s-inputs' % label])

            except:
                print('Failed to add %s. Skipping.' % patch_path)
        data = np.concatenate(data)
        fname = '/home/marysia/data/thesis/patches/lidc-localization-patches/%s_patches.npz' % label
        np.savez(fname, data=data, meta=metadata, scans=scans)


def extract_lidc_samples(label, folder, patch_folder):
    # TODO: subset only contains train samples, no test samples
    patches = glob.glob(patch_folder + '*.pkl.zip')

    print('Extracting %s samples from %d patches.' % (label, len(patches)))
    # initialize lists
    train_scans = []
    train_metadata = []
    train_data = []

    test_scans = []
    test_metadata = []
    test_data = []

    # loop through all patch-?.pkl.zips and extract positive samples
    for i, patch_path in enumerate(patches[0:5]):
        try:
            print('%d/%d: %s' % (i + 1, len(patches), patch_path))
            # loading only loads the keys, therefore takes little time
            unzipped = load(patch_path)

            # train:
            train_scans += unzipped['train-%s-scans' % label]
            train_metadata += unzipped['train-%s-metadata' % label]
            train_data.append(unzipped['train-%s-inputs' % label])

            print(train_data[-1].shape)
            # test:
            test_scans += unzipped['test-%s-scans' % label]
            test_metadata += unzipped['test-%s-metadata' % label]
            test_data.append(unzipped['test-%s-inputs' % label])


        except:
            print('Failed to add %s. Skipping.' % patch_path)

    # concatenate all samples (assuming same size)
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)
    print(test_data.shape)
    # save
    np.savez('/home/marysia/data/thesis/%s/%s_train_patches.npz' % (folder, label), data=train_data,
             meta=train_metadata, scans=train_scans)
    # np.savez('/home/marysia/data/thesis/%s/%s_test_patches.npz' % (folder, label), data=test_data, meta=test_metadata, scans=test_scans)


def extract_samples(label, folder, patch_folder):
    patches = glob.glob(patch_folder + '*.pkl.zip')

    print('Extracting %s samples from %d patches.' % (label, len(patches)))
    # initialize lists
    train = []
    test = []

    # loop through all patch-?.pkl.zips and extract positive samples
    for i, patch_path in enumerate(patches[0:2]):
        try:
            print('%d/%d: %s' % (i + 1, len(patches), patch_path))
            # loading only loads the keys, therefore takes little time
            unzipped = load(patch_path)
            print(unzipped.keys)
            # load the positive training and test samples
            # pos_train = unzipped['train-%s-inputs' % label]
            pos_test = unzipped['test-%s-inputs' % label]

            # append positive training and test samples to list
            # train.append(pos_train)
            test.append(pos_test)
        except:
            print('Failed to add %s. Skipping.' % patch_path)

    # concatenate all samples (assuming same size)
    # train = np.concatenate(train)
    test = np.concatenate(test)

    # save
    # np.savez('/home/marysia/data/thesis/%s/%s_all_train_patches.npz' % (folder, label), data=train)
    # np.savez('/home/marysia/data/thesis/%s/%s_all_test_patches.npz' % (folder, label), data=test)


def extract_positive_samples():
    """
    Extracts all available positive train and test samples from the zipped pickle files. Returns the amount of
    samples were extracted for the train and test set respectively.

    Returns:
        train_samples: train.shape[0], number of samples in the positive train set.
        test_samples: test.shape[0], number of samples in the positive test set.
    """
    print('Extracting positive samples from %d patches.' % len(patches))
    # initialize lists
    train = []
    test = []

    # loop through all patch-?.pkl.zips and extract positive samples
    for patch_path in patches:
        print(patch_path)
        # loading only loads the keys, therefore takes little time
        unzipped = load(patch_path)

        # load the positive training and test samples
        pos_train = unzipped['train-positive-inputs']
        pos_test = unzipped['test-positive-inputs']

        # append positive training and test samples to list
        train.append(pos_train)
        test.append(pos_test)

    # concatenate all samples (assuming same size)
    train = np.concatenate(train)
    test = np.concatenate(test)

    # save
    np.savez('/home/marysia/data/thesis/patches/positive_train_patches.npz', data=train)
    np.savez('/home/marysia/data/thesis/patches/positive_test_patches.npz', data=test)

    return train.shape[0], test.shape[0]


def extract_negative_samples(scope, samples):
    """
    Extracts a given number (samples) of negative samples from the zipped pickles, for the given scope (train or test).
    It does this by opening and reading in the zipped pickle patch while the amount of samples is not yet met, and
    concatenating the candidates to a list of candidates.

    Args:
        scope: str, train or test.
        samples: int, the amount of samples that will be extracted from the zipped pickles.
    """
    print('Extracting %d negative samples for the %s set.' % (samples, scope))
    # initialize lists
    li = []

    # initialize stop criteria
    limit = False
    i = 0
    total = 0

    # loop through while limit is not matched and still patch paths left
    while not limit and i < len(patches):
        # unzip and load negative inputs
        unzipped = load(patches[i])
        neg_inputs = unzipped[scope + '-negative-inputs']
        i += 1

        # add carelessly when adding will not exceed the sample limit
        if (total + neg_inputs.shape[0]) < samples:
            li.append(neg_inputs)
            total += neg_inputs.shape[0]
        # add portion of the candidates if adding all would exceed the sample limit
        else:
            to_add = samples - total
            li.append(neg_inputs[:to_add])
            total += to_add
            limit = True

    # save
    data = np.concatenate(li)
    np.savez('/home/marysia/data/thesis/patches/negative_' + scope + '_patches.npz', data=data)


# execute all steps
def balanced():
    train_samples, test_samples = extract_positive_samples()
    # extract_negative_samples('train', train_samples)
    # extract_negative_samples('test', test_samples)


def all():
    patch_folder = '/home/marysia/data/thesis/nlst-patches-1.3-3.5-annotated/'
    extract_samples('negative', 'patches', patch_folder)
    extract_samples('positive', 'patches')


def lidc():
    patch_folder = '/home/marysia/data/thesis/lidc/lidc-fp-reduction-subset-0/'
    extract_lidc_samples('negative', 'lidc-patches', patch_folder)
    extract_lidc_samples('positive', 'lidc-patches', patch_folder)


def lidc_localization():
    patch_folder = '/home/marysia/data/thesis/zipped_pickles/lidc-localization-model/'
    extract_lidc_localization_samples(patch_folder)


lidc_localization()