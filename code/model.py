import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Reshape, Input, Lambda, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D, AveragePooling3D
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

class Fully3D():
    def __init__(self, input_shape=(7, 72, 72, 1), base=16, blocks=4):
        self.model = Sequential()
        self.name = 'Fully_3D'
        self.base = base
        self.blocks = blocks
        self.input_shape = input_shape
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
        self.model.add(BatchNormalization())

    def end_block(self):
        self.model.add(Flatten())
        # self.model.add(Dense(512, init='normal'))
        # self.model.add(Activation('relu'))
        self.model.add(Dropout(.5))
        self.model.add(Dense(2, init='normal'))
        self.model.add(Activation('softmax'))

    def construct(self):
        filters = [self.base * (2 ** i) for i in range(self.blocks)]
        self.block(filters[0], first=True)
        for filter_nrs in filters[1:]:
            self.block(filter_nrs)
        self.end_block()

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



class Resnet3D():

    def __init__(self, input_shape=(7, 72, 72, 1), nb_classes=2, blocks=1):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.order = 'tf' if input_shape[3] == 1 else 'th'
        self.blocks = blocks
        self.name = 'Resnet3D'
        self.model = self.resnet()


    def conv_bn_relu(self, x, filter, subsample=(1, 1, 1)):
        ''' Convolution -> batch normalization -> relu '''
        x = Convolution3D(filter, 3, 3, 3, subsample=subsample, border_mode='same')(x)
        x = BatchNormalization(mode=0, axis=1)(x)
        x = Activation('relu')(x)
        return x

    def bn_relu_conv(self, x, filter, subsample=(1, 1, 1)):
        ''' Batch Normalization -> Activation -> Convolution'''
        x = BatchNormalization(mode=0, axis=1)(x)
        x = Activation('relu')(x)
        x = Convolution3D(filter, 3, 3, 3, border_mode='same', subsample=subsample)(x)
        return x

    def block_function(self, input, nb_filters, subsample=(1, 1, 1)):
        ''' Block function: two bn_relu_conv's and merge '''
        x = self.bn_relu_conv(input, nb_filters, subsample=subsample)
        x = self.bn_relu_conv(x, nb_filters)
        merged = self.merge_function(input, x)

        return merged

    def merge_function(self, input, residual):
        ''' Addition function '''
        dim1 = round(input._keras_shape[1] / float(residual._keras_shape[1]))
        dim2 = round(input._keras_shape[2] / float(residual._keras_shape[2]))
        dim3 = round(input._keras_shape[3] / float(residual._keras_shape[3]))
        equal_channels = residual._keras_shape[4] == input._keras_shape[4]

        if dim1 > 1 or dim2 > 1 or dim3 > 1 or not equal_channels:
            subsample = (dim2, dim3, dim1) if self.order == 'tf' else (dim1, dim2, dim3)
            filter = residual._keras_shape[4]

            input = Convolution3D(filter, 1, 1, 1, subsample=subsample, border_mode='valid')(input)
            input = Activation('relu')(input)

        merged_model = merge([input, residual], mode='sum')
        merged_model = Activation('relu')(merged_model)
        return merged_model

    def residual_block(self, x, filters, first=False, subsample=False):
        ''' Adds a self.blocks residual blocks.'''
        for i in range(self.blocks):
            subsample_vals = (1, 1, 1)
            if i == 0 and (first or subsample):
                subsample_vals = (2, 2, 2)
            x = self.block_function(x, nb_filters=filters, subsample=subsample_vals)
        return x

    def resnet(self):
        # first stuff

        input = Input(shape=self.input_shape)

        x = self.conv_bn_relu(input, 16)

        # # middle stuff
        x = self.residual_block(x, filters=16, first=True)
        x = self.residual_block(x, filters=32)
        x = self.residual_block(x, filters=64, subsample=True)
        x = self.residual_block(x, filters=128, subsample=True)

        # end stuff
        if self.order == 'tf':
            pool_size = (x._keras_shape[1], x._keras_shape[2], x._keras_shape[3])
            x = AveragePooling3D(pool_size=pool_size)(x)
        else:
            pool_size = (x._keras_shape[2], x._keras_shape[3], x._keras_shape[4])


        x = Flatten()(x)
        x = Dense(output_dim=self.nb_classes, init="he_normal", activation="softmax")(x)

        model = Model(input=input, output=x)
        return model



        
        
        