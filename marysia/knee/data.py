import random
import numpy as np

knee_data = '/home/ubuntu/knee/meniscus_tear/train_data.npz'
datadir = '/home/marysia/thesis/data'
knee_directory = '/home/marysia/thesis/data/KNEE/'

print('Starting..')
# read in
data = np.load(knee_data)
cor = data['cor']
sag = data['sag']
print('Loaded.')
# define data
cor = cor[:, 6:10, 1:101, 20:220]
sag = sag[:, 10:14, 1:, 7:207]

cor = cor.reshape(-1, 4, 100, 200)
sag = sag.reshape(-1, 4, 100, 200)
print('Reshaped.')

label_cor = np.ones(cor.shape[0])
label_sag = np.zeros(sag.shape[0])
print('Labeled.')

# combine and shuffle
all_data = np.array(np.concatenate([sag, cor]))
all_labels = np.array(np.concatenate([label_sag, label_cor]))

zipped = list(zip(all_data, all_labels))
random.shuffle(zipped)
all_data, all_labels = zip(*zipped)
print('Shuffled.')

# define training, validation and test sets
train_length = 15000
val_length = 20000

train_data = all_data[:train_length]
train_labels = all_labels[:train_length]

val_data = all_data[train_length:val_length]
val_labels = all_labels[train_length:val_length]

test_data = all_data[val_length:]
test_labels = all_labels[val_length:]
print('Cut into training, val and test set.')

np.savez(knee_directory + '3dtrain_all.npz', data=all_data, labels=all_labels)
np.savez(knee_directory + '3dtrain.npz', data=train_data, labels=train_labels)
np.savez(knee_directory + '3dvalid.npz', data=val_data, labels=val_labels)
np.savez(knee_directory + '3dtest.npz', data=test_data, labels=test_labels)
print('Saved!')




