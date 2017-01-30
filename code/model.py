import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Reshape, Input, Lambda, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization

class Zuidhof():
    def __init__(self, input_shape=(1, 7, 72, 72), base=16, blocks=4):
        self.input_shape = input_shape
        self.base = base
        self.blocks = blocks
        self.inputs = Input(shape=input_shape)
        self.x = self.inputs # a tensor
        self.model = self.construct()
        self.name = 'Zuidhof CNN'
        self.description = '3D convolutional network based on Zuidhof\'s implementation for lung nodules. This version is not a resnet.'


    def start_block(self, filters, activation='relu', border_mode='same', dropout=.3):
        self.x = Convolution3D(filters, 1, 3, 3, input_shape=self.input_shape, border_mode=border_mode)(self.x)
        self.x = BatchNormalization()(self.x)
        self.x = Activation(activation)(self.x)
        self.x = Dropout(dropout)(self.x)
        self.x = Convolution3D(filters, 1, 3, 3, border_mode=border_mode)(self.x)

    def mid_block(self, filters, activation='relu', border_mode='same', dropout=.3):
        self.x = BatchNormalization()(self.x)
        self.x = Activation(activation)(self.x)
        self.x = Convolution3D(filters, 1, 3, 3, subsample=(2, 2, 2), border_mode=border_mode)(self.x)
        self.x = BatchNormalization()(self.x)
        self.x = Activation(activation)(self.x)
        self.x = Dropout(.3)(self.x)
        self.x = Convolution3D(filters, 1, 3, 3, border_mode=border_mode)(self.x)

    def end_block(self, activation='relu', final_activation='softmax'):
        self.x = Flatten()(self.x)
        self.x = Dense(512)(self.x)
        self.x = Activation(activation)(self.x)
        self.x = Dense(2)(self.x)
        self.x = Activation(final_activation)(self.x)

    def construct(self):
        filters = [self.base * (2 ** i) for i in range(self.blocks)]
        # initial block
        self.start_block(filters[0])
        # middle blocks
        for filter_nrs in filters[1:]:
            self.mid_block(filter_nrs)
        self.end_block()

        return Model(input=self.inputs, output=self.x)


class CNN_3D():
    def __init__(self, input_shape=(1, 7, 72, 72), base=16, blocks=4):
        self.model = Sequential()
        self.base = base
        self.blocks = blocks
        self.input_shape = input_shape
        self.name = 'CNN_3D'
        self.description = '3D convolutional network with input shape ' + str(self.input_shape) + ', ' + str(self.blocks) + ' blocks and starting at ' + str(self.base) + ' filters.'
        self.set_description()
        self.construct()
        
    # --- blocks for model construction --- #     
    def start_block(self, filters, border_mode='same', activation='relu', dropout=.3): 
        self.model.add(Convolution3D(filters, 1, 3, 3, border_mode=border_mode, input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation))
        self.model.add(Dropout(dropout))
        self.model.add(Convolution3D(filters, 1, 3, 3, border_mode=border_mode))
        
    def mid_block(self, filters, subsample=(2,2,2), border_mode='same', dropout=.3, activation='relu'): 
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation))
        self.model.add(Convolution3D(filters, 1, 3, 3, subsample=subsample, border_mode=border_mode))
        self.model.add(BatchNormalization())
        self.model.add(Activation(activation))
        self.model.add(Dropout(dropout))
        self.model.add(Convolution3D(filters, 1, 3, 3, border_mode=border_mode))
        
    def end_block(self, init='normal', activation='relu', final_activation='softmax', dropout=.5):
        self.model.add(Flatten())
        self.model.add(Dense(512, init=init))
        self.model.add(Activation(activation))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(2, init=init))
        self.model.add(Activation(final_activation))
    
    def construct(self):
        filters = [self.base*(2**i) for i in range(self.blocks)]
        # initial block 
        self.start_block(filters[0])
        # middle blocks 
        for filter_nrs in filters[1:]:
            self.mid_block(filter_nrs)
        self.end_block()
    
    # --- helper functions --- #
    def get_model(self):
        return self.model
    
    def set_description(self): 
        string = '3D convolutional network. Input shape: '
        string += str(self.input_shape)
        string += ' Blocks: '
        string += str(self.blocks)
        string += ' Filters: '
        string += str([self.base*(2**i) for i in range(self.blocks)])
        
        self.description = string

        
        
        