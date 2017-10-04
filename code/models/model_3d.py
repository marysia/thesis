import util.base_layers as base
import util.gconv_layers as gconv
import util.layers as layer
from basemodel import BaseModel


class CNN(BaseModel):
    def build_graph(self):
        num_filters = [16, 32, 64]
        self.activation = 'relu'

        tensor = self.conv_bn_act(self.x, num_filters[0], first=True)
        tensor = base.maxpool3d(tensor, strides=[1, 1, 2, 2, 1])

        tensor = self.conv_bn_act(tensor, num_filters[0])
        # tensor = base.dropout(tensor, keep_prob=.7, training=self.training)

        tensor = self.conv_bn_act(tensor, num_filters[1])
        tensor = base.maxpool3d(tensor, strides=[1, 2, 2, 2, 1])

        tensor = self.conv_bn_act(tensor, num_filters[1])
        # tensor = base.dropout(tensor, keep_prob=.7, training=self.training)


        tensor = self.conv_bn_act(tensor, num_filters[2])
        tensor = base.maxpool3d(tensor, strides=[1, 2, 2, 2, 1])

        tensor = self.conv_bn_act(tensor, num_filters[2])

        tensor = base.flatten(tensor)
        final = base.readout(tensor, [int(tensor.get_shape()[-1]), self.data.nb_classes])
        return final

    def conv_bn_act(self, tensor, filters, first=False):
        if self.group == 'Z3':
            tensor = base.convolution3d(tensor, [3, 3, 3], filters)
        else:
            in_group = 'Z3' if first else self.group
            tensor = gconv.gconvolution3d(tensor, in_group, self.group, filters)

        tensor = base.batch_normalization(tensor)
        tensor = base.activation(tensor, key=self.activation)
        return tensor

class WideBoostingNetwork(BaseModel):
    def build_graph(self):
        k = 1
        num_filters = [8 * k, 16 * k, 32 * k, 64 * k]
        self.activation = 'relu'

        tensor = self.x
        tensor = self.convolution3d(tensor, 16, (3, 3, 3), (1, 1, 1), first=True)
        tensor = self.resnet_block(tensor, num_filters[0], self.activation, length=3, strided=False)
        tensor = self.resnet_block(tensor, num_filters[1], self.activation, length=3, strided=True)
        tensor = self.resnet_block(tensor, num_filters[2], self.activation, length=3, strided=True)
        tensor = self.resnet_block(tensor, num_filters[3], self.activation, length=3, strided=True)

        tensor = base.batch_normalization(tensor)
        tensor = base.activation(tensor, key=self.activation)

        # global average pooling instead
        tensor = base.maxpool3d(tensor, strides=[1, 2, 9, 9, 1])

        tensor = base.convolution3d(tensor, nb_channels_out=2, filter_shape=(1, 1, 1))

        # just sigmoid or softmax instead
        tensor = base.flatten(tensor)
        final = base.readout(tensor, [int(tensor.get_shape()[-1]), self.data.nb_classes])
        return final

    def resnet_block(self, tensor, filters, activation, length=3, strided=True):

        for i in xrange(length):
            save_tensor = tensor

            stride = (2, 2, 2) if i == 0 and strided else (1, 1, 1)

            tensor = base.batch_normalization(tensor)
            tensor = base.activation(tensor, activation)
            tensor = self.convolution3d(tensor, filters, (3, 3, 3), (1, 1, 1))  # CHANGED

            tensor = base.batch_normalization(tensor)
            tensor = base.activation(tensor, activation)
            # dropout
            tensor = self.convolution3d(tensor, filters, (3, 3, 3), stride)

            if i == 0:
                save_tensor = self.convolution3d(save_tensor, filters, (1, 1, 1), stride)

            tensor = base.merge(save_tensor, tensor, method='add')
        return tensor

    def convolution3d(self, tensor, filters, filter_shape, stride, first=False):
        if self.group == 'Z3':
            return base.convolution3d(tensor, filter_shape, filters, stride)
        else:
            in_group = 'Z3' if first else self.group
            return gconv.gconvolution3d(tensor, in_group, self.group, filters)