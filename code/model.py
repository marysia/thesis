import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Reshape, Input, Lambda, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization

class CNN_3D():
    def __init__(self, input_shape=(7, 72, 72, 1), base=16, blocks=4): 
        self.model = Sequential()
        self.base = base
        self.blocks = blocks
        self.input_shape = input_shape
        self.model_name = 'CNN_3D'
        self.model_description = '3D convolutional network'
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
    
    def get_model(self):
        return self.model
        

        
        
        