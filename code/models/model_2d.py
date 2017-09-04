import tensorflow as tf

import util.base_layers as base_layer
import util.gconv_layers as gconv
import util.layers as layer
from basemodel import BaseModel


class Z2CNN(BaseModel):
    def build_graph(self):
        # l1 and l2
        tensor = layer.conv2d_bn_act(self.x, nb_channels_out=20)
        # tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=20)

        # max pooling
        tensor = tf.nn.max_pool(tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # l3
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=20)
        # tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)
        # l4
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=20)
        # tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)
        # l5
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=20)
        # tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)
        # l6
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=20)
        # tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)

        # top
        tensor = base_layer.convolution2d(tensor, filter_shape=[4, 4], nb_channels_out=10)

        tensor = base_layer.dense(tensor, 512)
        tensor = base_layer.activation(tensor, key='relu')
        final = base_layer.readout(tensor, [512, self.data.nb_classes])

        self.model_logits = final


class P4CNN(BaseModel):
    def build_graph(self):
        group = 'C4'
        # l1 and l2
        tensor = gconv.gconv_bn_act(self.x, in_group='Z2', out_group=group, out_channels=20)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group, out_channels=20)

        # max pooling
        tensor = tf.nn.max_pool(tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # l3-l6
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group, out_channels=20)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group, out_channels=20)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group, out_channels=20)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group, out_channels=20)

        tensor = gconv.gconv_wrapper2d(tensor, in_group=group, out_group=group, ksize=4, in_channels=20,
                                       out_channels=10)

        tensor = base_layer.dense(tensor, 256)
        tensor = base_layer.activation(tensor, key='relu')
        final = base_layer.readout(tensor, [256, self.data.nb_classes])

        self.model_logits = final


class P4CNNDropout(BaseModel):
    def build_graph(self):
        group = 'C4'
        # l1 and l2
        tensor = gconv.gconv_bn_act(self.x, in_group='Z2', out_group=group, out_channels=20)
        tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group, out_channels=20)

        # max pooling
        tensor = tf.nn.max_pool(tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # l3-l6
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group, out_channels=20)
        tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group, out_channels=20)
        tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group, out_channels=20)
        tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group, out_channels=20)
        tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)

        tensor = gconv.gconv_wrapper2d(tensor, in_group=group, out_group=group, ksize=4, in_channels=20,
                                       out_channels=10)

        tensor = base_layer.dense(tensor, 256)
        tensor = base_layer.activation(tensor, key='relu')
        final = base_layer.readout(tensor, [256, self.data.nb_classes])

        self.model_logits = final


class ConvolutionalModel1(BaseModel):
    def build_graph(self):
        # # layers
        tensor = layer.conv2d_bn_act(self.x, nb_channels_out=16)
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=16)
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=16)
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=16)
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=16)
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=16)
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=16)
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=16)
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=16)
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=16)
        tensor = layer.conv2d_bn_act(tensor, nb_channels_out=16)

        tensor = base_layer.dense(tensor, 512)
        tensor = base_layer.activation(tensor, key='relu')
        final = base_layer.readout(tensor, [512, self.data.nb_classes])

        self.model_logits = final


class Resnet(BaseModel):
    def build_graph(self):
        blocks = 3

        tensor = layer.conv2d_bn_act(self.x)

        for _ in xrange(blocks):
            tensor = self.residual_block(tensor)

        tensor = base_layer.dense(tensor, 512)
        tensor = base_layer.activation(tensor, key='relu')
        final = base_layer.readout(tensor, [512, self.data.nb_classes])

        self.model_logits = final

    def residual_block(self, tensor_in):
        tensor = layer.act_bn_conv2d(tensor_in)
        tensor = base_layer.dropout(tensor, keep_prob=0.8, training=self.training)
        tensor = layer.act_bn_conv2d(tensor)
        tensor = base_layer.dropout(tensor, keep_prob=0.8, training=self.training)
        tensor_out = base_layer.merge(tensor_in, tensor, method='add')
        return tensor_out


class GResnet(BaseModel):
    def build_graph(self):
        blocks = 3
        in_group, out_group = ('Z2', 'C4')

        tensor = gconv.gconv_bn_act(self.x, in_group=in_group, out_group=out_group)

        for i in xrange(blocks):
            tensor = self.residual_block(tensor, out_group, out_group)

        tensor = base_layer.dense(tensor, 512)
        tensor = base_layer.activation(tensor, key='relu')
        final = base_layer.readout(tensor, [512, self.data.nb_classes])

        self.model_logits = final

    def residual_block(self, tensor_in, in_group, out_group):
        tensor = gconv.act_bn_gconv(tensor_in, in_group=in_group, out_group=out_group)
        tensor = base_layer.dropout(tensor, keep_prob=0.8, training=self.training)
        tensor = gconv.act_bn_gconv(tensor, in_group=out_group, out_group=out_group)
        tensor = base_layer.dropout(tensor, keep_prob=0.8, training=self.training)
        tensor_out = base_layer.merge(tensor_in, tensor, method='add')

        return tensor_out


class GConvModel1(BaseModel):
    def build_graph(self):
        group = 'C4'
        # group = 'D4'
        tensor = gconv.gconv_bn_act(self.x, in_group='Z2', out_group=group)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group)
        tensor = gconv.gconv_bn_act(tensor, in_group=group, out_group=group)
        tensor = base_layer.dense(tensor, 256)
        tensor = base_layer.activation(tensor, key='relu')
        final = base_layer.readout(tensor, [256, self.data.nb_classes])

        self.model_logits = final
