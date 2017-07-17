from basemodel import BaseModel
import util.layers as layer
import util.base_layers as base_layer
import util.gconv_layers as gconv

class Z3CNN(BaseModel):
    def build_graph(self):
        nb_out_channels = 20
        # l1 and l2
        tensor = layer.conv3d_bn_act(self.x, nb_channels_out=nb_out_channels)
        #tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)
        tensor = layer.conv3d_bn_act(tensor, nb_channels_out=nb_out_channels)
        #
        # # l3
        # tensor = layer.conv3d_bn_act(tensor, nb_channels_out=nb_out_channels)
        # tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)
        # # l4
        # tensor = layer.conv3d_bn_act(tensor, nb_channels_out=nb_out_channels)
        # tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)
        # # l5
        # tensor = layer.conv3d_bn_act(tensor, nb_channels_out=nb_out_channels)
        # tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)
        # # l6
        # tensor = layer.conv3d_bn_act(tensor, nb_channels_out=nb_out_channels)
        # tensor = base_layer.dropout(tensor, keep_prob=.7, training=self.training)
        #
        # # top
        # tensor = base_layer.convolution3d(tensor, filter_shape=[4, 4, 4], nb_channels_out=10)

        tensor = base_layer.dense(tensor, 256)
        tensor = base_layer.activation(tensor, key='relu')
        final = base_layer.readout(tensor, [256, self.data.nb_classes])

        self.model_logits = final

class GCNN(BaseModel):
    def build_graph(self):
        group = 'O'
        nb_channels_out = 10
        # l1 and l2
        tensor = gconv.gconv3d_bn_act(self.x, in_group='Z3', out_group=group, out_channels=nb_channels_out)
        tensor = gconv.gconv3d_bn_act(tensor, in_group=group, out_group=group, out_channels=nb_channels_out)

        tensor = base_layer.dense(tensor, 256)
        tensor = base_layer.activation(tensor, key='relu')
        final = base_layer.readout(tensor, [256, self.data.nb_classes])

        self.model_logits = final

class Resnet(BaseModel):
    def build_graph(self):
        nb_out_channels = 16
        blocks = 3
        tensor = layer.conv3d_bn_act(self.x, nb_channels_out=nb_out_channels)

        for i in xrange(blocks):
            tensor = self.residual_block(tensor, nb_out_channels)

        tensor = base_layer.dense(tensor, 256)
        tensor = base_layer.activation(tensor, key='relu')
        final = base_layer.readout(tensor, [256, self.data.nb_classes])

        self.model_logits = final


    def residual_block(self, input_tensor, nb_channels_out):
        tensor = layer.bn_act_conv3d(input_tensor, nb_channels_out=nb_channels_out)
        tensor = layer.bn_act_conv3d(tensor, nb_channels_out=nb_channels_out)
        tensor = base_layer.merge(input_tensor, tensor, method='add')
        return tensor

class GResnet(BaseModel):
    def build_graph(self):
        nb_out_channels = 16
        blocks = 3
        tensor = layer.gconv3d_bn_act(self.x, nb_channels_out=nb_out_channels)

        for i in xrange(blocks):
            tensor = self.residual_block(tensor, nb_out_channels)

        tensor = base_layer.dense(tensor, 256)
        tensor = base_layer.activation(tensor, key='relu')
        final = base_layer.readout(tensor, [256, self.data.nb_classes])

        self.model_logits = final

    def residual_block(self, input_tensor, nb_channels_out):
        tensor = layer.bn_act_gconv3d(input_tensor, nb_channels_out=nb_channels_out)
        tensor = layer.bn_act_gconv3d(tensor, nb_channels_out=nb_channels_out)
        tensor = base_layer.merge(input_tensor, tensor, method='add')
        return tensor