import tensorflow as tf
import sys
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)

from keras.datasets import cifar10
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)

from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Lambda
)
from keras.utils import np_utils
import keras.backend as K
K.set_image_dim_ordering('th')


def conv_bn_relu(input, nb_filter, nb_row, nb_col, wr, subsample=(1, 1)):
    conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                         init="he_normal", border_mode="same", W_regularizer=wr)(input)
    norm = BatchNormalization(mode=0, axis=1)(conv)
    return Activation("relu")(norm)


def bn_relu_conv(input, nb_filter, nb_row, nb_col, W_regularizer, subsample=(1, 1)):
    norm = BatchNormalization(mode=0, axis=1)(input)
    activation = Activation("relu")(norm)
    return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                         init="he_normal", border_mode="same", W_regularizer=W_regularizer)(activation)


def block_function(input, nb_filters, init_subsample=(1, 1)):
    conv_1_1 = bn_relu_conv(input, nb_filters, 3, 3, W_regularizer=l2(weight_decay), subsample=init_subsample)
    conv_3_3 = bn_relu_conv(conv_1_1, nb_filters, 3, 3, W_regularizer=l2(weight_decay))
    return shortcut(input, conv_3_3)


def shortcut(input, residual):
    stride_width = input._keras_shape[2] / residual._keras_shape[2]
    stride_height = input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        ss = (stride_height, stride_width)
        f = residual._keras_shape[1]  # orig

        shortcut = Convolution2D(nb_filter=f, nb_row=1, nb_col=1, subsample=ss,
                                 init="he_normal", border_mode="valid", W_regularizer=l2(weight_decay))(input)
        shortcut = Activation("relu")(shortcut)

    M1 = merge([shortcut, residual], mode="sum")
    M1 = Activation("relu")(M1)
    return M1


# Builds a residual block with repeating bottleneck blocks.
def residual_block(input, nb_filters, repetations, is_first_layer=False, subsample=False):
    for i in range(repetations):
        init_subsample = (1, 1)
        if i == 0 and (is_first_layer or subsample):
            init_subsample = (2, 2)
        input = block_function(input, nb_filters=nb_filters, init_subsample=init_subsample)

    return input

# constants
learning_rate = 0.01
momentum = 0.9
img_rows, img_cols = 32, 32
img_channels = 3
nb_epochs = 400
batch_size = 700
nb_classes = 10
pL = 0.5
weight_decay = 1e-4

def resnet():
    input = Input(shape=(img_channels, img_rows, img_cols))
    conv1 = conv_bn_relu(input, nb_filter=16, nb_row=3, nb_col=3, wr=l2(weight_decay))
    # middle stuff
    block1 = residual_block(conv1, nb_filters=16, repetations=2, is_first_layer=True)
    block2 = residual_block(block1, nb_filters=32, repetations=2)
    block3 = residual_block(block2, nb_filters=64, repetations=2, subsample=True)
    # Classifier block
    pool2 = AveragePooling2D(pool_size=(8, 8))(block3)
    flatten1 = Flatten()(pool2)
    final = Dense(output_dim=10, init="he_normal", activation="softmax", W_regularizer=l2(weight_decay))(flatten1)

    model = Model(input=input, output=final)
    print 'Model done.'
    return model


# data
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
Y_train = np_utils.to_categorical(Y_train, nb_classes)
X_test = X_test.astype('float32')
Y_test = np_utils.to_categorical(Y_test, nb_classes)


nb_epochs = 1
batch_size = 700

from logger import Logger
from control import ProgramEnder

ender = ProgramEnder()
log = Logger(sys.argv)
# building and training net
model = resnet()
log.info('Resnet created.')
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
log.info('Model compiled.')
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=nb_epochs, batch_size=batch_size)
log.info('Training done.')
scores = model.evaluate(X_test, Y_test)
log.result(str(round(scores[1]*100, 3)))
log.copy()

