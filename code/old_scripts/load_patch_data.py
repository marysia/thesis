import os
import numpy as np
from code.preprocessing.generic import data_metrics
from code.preprocessing.visualize import array_to_mp4, array_to_png_folder
from code.preprocessing.patches import DataPatches
from code.preprocessing.mnist import DataMNIST

data = DataMNIST()
data_metrics(data)

data = DataPatches()
data_metrics(data)

# create images to visuale the nodules
video_folder = '/home/marysia/thesis/results/video'
for i in xrange(5):
    # folder name in i_train_label format.
    folder_name = os.path.join(video_folder, str(i) + '_train_' + str(np.argmax(data.train.y[i])))
    array_to_png_folder(data.train.x[i], folder_name)
