import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Reshape, Input, Lambda, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D, AveragePooling3D
from keras.layers.normalization import BatchNormalization

class Zuidhof():

    def __init__(self, input_shape=(7, 72, 72, 1), base=16, blocks=4):
        # initialise parameters
        self.input_shape = input_shape
        self.base = base
        self.blocks = blocks
        
        # construct 
        self.inputs = Input(shape=input_shape)
        self.model = self.construct()
        self.name = 'ZuidhofCNN'
     

    # --- Conv -> BN -> Act -> DO -> Conv --- #
    def start_block(self, x, filters):
        x = Convolution3D(filters, 3, 3, 3, input_shape=self.input_shape, border_mode='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(.3)(x)
        x = Convolution3D(filters, 3, 3, 3, border_mode='same')(x)
        return x

    # --- BN -> Act -> Conv -> BN -> Act -> DO -> Conv --- #
    def mid_block(self, x, filters):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution3D(filters, 3, 3, 3, subsample=(2, 2, 2), border_mode='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(.3)(x)
        x = Convolution3D(filters, 3, 3, 3, border_mode='same')(x)
        return x

    # --- Flatten
    def end_block(self, x):
        x = Flatten()(x)
        x = Dense(2)(x)
        x = Activation('softmax')(x)
        return x

    def construct(self):
        filters = [self.base * (2 ** i) for i in range(self.blocks)]
        # initial block
        x = self.start_block(self.inputs, filters[0])
        # middle blocks
        for filter_nrs in filters[1:]:
            x = self.mid_block(x, filter_nrs)
        x = self.end_block(x)

        return Model(input=self.inputs, output=x)

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




class ZuidhofRN():

    def __init__(self, input_shape=(7, 72, 72, 1), blocks=1):
        # initialize parameters
        self.input_shape = input_shape
        self.order = 'tf' if input_shape[3] == 1 else 'th'
        self.blocks = blocks
        self.name = 'ZuidhofResnet3D'
        self.model = self.construct()



    # --- shortcut functions --- #
    def conv_bn_act(self, x, filters, subsample=(1, 1, 1)):
        x = Convolution3D(filters, 3, 3, 3, subsample=subsample, border_mode='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    def bn_act(self, x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    def dp_conv(self, x, filters):
        x = Dropout(.3)(x)
        x = Convolution3D(filters, 3, 3, 3, border_mode='same')(x)
        return x
    # --- merge --- #
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
        return merged_model

    # --- start layers --- #
    def start_block(self, input):
        # set x: conv_bn_act  with filters=16
        x = self.conv_bn_act(input, 16)

        # set y: conv-bn-act-dp-conv with filters=32
        y = self.conv_bn_act(x, 32)
        y = self.dp_conv(y, 32)

        # extra convolution on x
        x = Convolution3D(32, 1, 1, 1)(x)

        merged = self.merge_function(x, y)
        return merged

    # --- create the residual block. --- #
    def residual_block(self, input, filters, subsample=(1, 1, 1)):
        x = self.bn_act(input)
        x = self.conv_bn_act(x, filters, subsample=subsample)
        x = self.dp_conv(x, filters)
        merged = self.merge_function(input, x)
        return merged

    # --- wrapper to add self.blocks residual blocks (only first gets subsampled) --- #
    def residual_wrapper(self, x, filters):
        for i in xrange(self.blocks):
            subsample = (1, 1, 1)
            if i == 0:
                subsample = (2, 2, 2)
            x = self.residual_block(x, filters, subsample)
        return x

    # --- constructs model --- #
    def construct(self):
        input = Input(shape=self.input_shape)
        x = self.start_block(input)
        x = self.residual_wrapper(x, 32)
        x = self.residual_wrapper(x, 64)
        x = self.residual_wrapper(x, 128)

        # end.
        x = Flatten()(x)
        x = Dense(2)(x)
        x = Activation('softmax')(x)
        model = Model(input=input, output=x)
        return model


    # first
    # case:
    # conv
    # bn
    # act
    # dropout
    # conv
    # --> add
    #
    # other
    # cases:
    # bn
    # act
    # conv
    # bn
    # act
    # dropout
    # conv
    # -->


    # --- Convolution -> BN -> Act --- #
    def conv_bn_relu(self, x, filter, subsample=(1, 1, 1)):
        x = Convolution3D(filter, 3, 3, 3, subsample=subsample, border_mode='same')(x)
        x = BatchNormalization(mode=0, axis=1)(x)
        x = Activation('relu')(x)
        return x

class Resnet3D():
    def __init__(self, input_shape=(7, 72, 72, 1), blocks=1):
        # initialize parameters
        self.input_shape = input_shape
        self.order = 'tf' if input_shape[3] == 1 else 'th'
        self.blocks = blocks
        self.name = 'Resnet3D'
        self.model = self.construct()

    # --- Convolution -> BN -> Act --- #
    def conv_bn_relu(self, x, filter, subsample=(1, 1, 1)):
        x = Convolution3D(filter, 3, 3, 3, subsample=subsample, border_mode='same')(x)
        x = BatchNormalization(mode=0, axis=1)(x)
        x = Activation('relu')(x)
        return x

    # --- BN -> Act --> Conv --- #
    def bn_relu_conv(self, x, filter, subsample=(1, 1, 1)):
        ''' Batch Normalization -> Activation -> Convolution'''
        x = BatchNormalization(mode=0, axis=1)(x)
        x = Activation('relu')(x)
        x = Convolution3D(filter, 3, 3, 3, border_mode='same', subsample=subsample)(x)
        return x

    # --- 2x BN->Act->Conv and merge --- #
    def block_function(self, input, nb_filters, subsample=(1, 1, 1)):
        ''' Block function: two bn_relu_conv's and merge '''
        x = self.bn_relu_conv(input, nb_filters, subsample=subsample)
        x = self.bn_relu_conv(x, nb_filters)
        merged = self.merge_function(input, x)
        return merged

    # --- Merge function: reshapes input if necessary --- #
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

    # --- Adds self.blocks number of residual blocks --- #
    def residual_block(self, x, filters, first=False, subsample=False):
        for i in range(self.blocks):
            subsample_vals = (1, 1, 1)
            if i == 0 and (first or subsample):
                subsample_vals = (2, 2, 2)
            x = self.block_function(x, nb_filters=filters, subsample=subsample_vals)
        return x


    # --- constructs resnet --- #
    def construct(self):
        input = Input(shape=self.input_shape)
        x = self.conv_bn_relu(input, 16)

        # middle stuff - residual blocks.
        x = self.residual_block(x, filters=16, first=True)
        x = self.residual_block(x, filters=32)
        x = self.residual_block(x, filters=64, subsample=True)
        x = self.residual_block(x, filters=128, subsample=True)

        # Pool if dim ordering = tf.
        if self.order == 'tf':
            pool_size = (x._keras_shape[1], x._keras_shape[2], x._keras_shape[3])
            x = AveragePooling3D(pool_size=pool_size)(x)

        # end.
        x = Flatten()(x)
        x = Dense(2)(x)
        x = Activation('softmax')(x)

        model = Model(input=input, output=x)
        return model



        
        
        