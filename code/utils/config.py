import os

# directories 
home_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
log_dir = os.path.join(home_dir, 'logs')

# working remotely: 
data_dir = '/zdev/data/'

# working locally: 
if not os.path.exists(data_dir):
    data_dir = os.path.join(home_dir, 'data')

# candidate info
patch_size = (7, 72, 72)
scale = (2.5, .512, .512)

# model info 
batch_size = 32
epochs = 50

optimized = 'sgd'
loss = 'binary_crossentropy'
metrics = ['accuracy']
lrate = .9
momentum = .9
decay = lrate / epochs
nesterov = False
