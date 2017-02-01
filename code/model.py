import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Reshape, Input, Lambda, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization

class Zuidhof():
    def __init__(self, input_shape=(7, 72, 72, 1), base=16, blocks=4, activation='relu', border_mode='same', dropout=.3):
        # initialise parameters
        self.input_shape = input_shape
        self.base = base
        self.blocks = blocks
        self.activation = activation
        self.border_mode = border_mode
        self.dropout = dropout
        
        # construct 
        self.inputs = Input(shape=input_shape)
        self.x = self.inputs # a tensor
        self.model = self.construct()
        self.name = 'ZuidhofCNN'
     

    def start_block(self, filters):
        self.x = Convolution3D(filters, 1, 3, 3, input_shape=self.input_shape, border_mode=self.border_mode)(self.x)
        self.x = BatchNormalization()(self.x)
        self.x = Activation(self.activation)(self.x)
        self.x = Dropout(self.dropout)(self.x)
        self.x = Convolution3D(filters, 1, 3, 3, border_mode=self.border_mode)(self.x)

    def mid_block(self, filters):
        self.x = BatchNormalization()(self.x)
        self.x = Activation(self.activation)(self.x)
        self.x = Convolution3D(filters, 1, 3, 3, subsample=(2, 2, 2), border_mode=self.border_mode)(self.x)
        self.x = BatchNormalization()(self.x)
        self.x = Activation(self.activation)(self.x)
        self.x = Dropout(self.dropout)(self.x)
        self.x = Convolution3D(filters, 1, 3, 3, border_mode=self.border_mode)(self.x)

    def end_block(self):
        self.x = Flatten()(self.x)
        self.x = Dense(512)(self.x)
        self.x = Activation(self.activation)(self.x)
        self.x = Dense(2)(self.x)
        self.x = Activation('softmax')(self.x)

    def construct(self):
        filters = [self.base * (2 ** i) for i in range(self.blocks)]
        # initial block
        self.start_block(filters[0])
        # middle blocks
        for filter_nrs in filters[1:]:
            self.mid_block(filter_nrs)
        self.end_block()

        return Model(input=self.inputs, output=self.x)


class fully3D():
    def __init__(self, input_shape=(7, 72, 72, 1), base=16, blocks=4):
        self.model = Sequential()
        self.base = base
        self.blocks = blocks
        self.input_shape = input_shape
        self.name = 'fully_3D'
        self.construct()


    def block(self, filters, first=False):
        if first:
            self.model.add(Convolution3D(filters, 1, 3, 3, border_mode='same', input_shape=self.input_shape))
        else:
            self.model.add(Convolution3D(filters, 1, 3, 3, border_mode='same', subsample=(2, 2, 2)))

        self.model.add(Activation('relu'))
        self.model.add(Convolution3D(filters, 1, 3, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(.3))
        #self.model.add(BatchNormalization())

    def end_block(self):
        self.model.add(Flatten())
        # self.model.add(Dense(512, init='normal'))
        # self.model.add(Activation('relu'))
        # self.model.add(Dropout(.5))
        self.model.add(Dense(2, init='normal'))
        self.model.add(Activation('softmax'))

    def construct(self):
        filters = [self.base * (2 ** i) for i in range(self.blocks)]
        self.block(filters[0], first=True)
        for filter_nrs in filters[1:]:
            self.block(filter_nrs)

class CNN_3D():
    def __init__(self, input_shape=(7, 72, 72, 1), base=16, blocks=4, activation='relu', border_mode='same', dropout=.3):
        # initialize parameters
        self.input_shape = input_shape
        self.base = base
        self.blocks = blocks
        self.activation = activation
        self.border_mode = border_mode
        self.dropout = dropout
        # construct
        self.model = Sequential()
        self.construct()
        self.name = 'CNN_3D'
        
    # --- blocks for model construction --- #     
    def start_block(self, filters):
        self.model.add(Convolution3D(filters, 1, 3, 3, border_mode=self.border_mode, input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation(self.activation))
        self.model.add(Dropout(self.dropout))
        self.model.add(Convolution3D(filters, 1, 3, 3, border_mode=self.border_mode))
        
    def mid_block(self, filters, subsample=(2,2,2)):
        self.model.add(BatchNormalization())
        self.model.add(Activation(self.activation))
        self.model.add(Convolution3D(filters, 1, 3, 3, subsample=subsample, border_mode=self.border_mode))
        self.model.add(BatchNormalization())
        self.model.add(Activation(self.activation))
        self.model.add(Dropout(self.dropout))
        self.model.add(Convolution3D(filters, 1, 3, 3, border_mode=self.border_mode))
        
    def end_block(self):
        self.model.add(Flatten())
        self.model.add(Dense(512, init='normal'))
        self.model.add(Activation(self.activation))
        self.model.add(Dropout(.5))
        self.model.add(Dense(2, init='normal'))
        self.model.add(Activation('softmax'))
    
    def construct(self):
        filters = [self.base*(2**i) for i in range(self.blocks)]
        # initial block 
        self.start_block(filters[0])
        # middle blocks 
        for filter_nrs in filters[1:]:
            self.mid_block(filter_nrs)
        self.end_block()


        
        
        