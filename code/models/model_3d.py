import util.base_layers as base
import util.gconv_layers as gconv
import util.layers as layer
from basemodel import BaseModel


class CNN(BaseModel):
    def build_graph(self):
        self.filters = [16, 16, 32, 32, 64, 64]

        tensor = self.conv_bn_act(self.x, 0)
        tensor = base.maxpool3d(tensor, strides=[1, 1, 2, 2, 1])

        tensor = self.conv_bn_act(tensor, 1)
        # tensor = base.dropout(tensor, keep_prob=.7, training=self.training)

        tensor = self.conv_bn_act(tensor, 2)
        tensor = base.maxpool3d(tensor, strides=[1, 2, 2, 2, 1])

        tensor = self.conv_bn_act(tensor, 3)
        # tensor = base.dropout(tensor, keep_prob=.7, training=self.training)


        tensor = self.conv_bn_act(tensor, 4)
        tensor = base.maxpool3d(tensor, strides=[1, 2, 2, 2, 1])

        tensor = self.conv_bn_act(tensor, 5)

        tensor = base.flatten(tensor)
        final = base.readout(tensor, [int(tensor.get_shape()[-1]), self.data.nb_classes])
        return final

    def conv_bn_act(self, tensor, i):
        raise NotImplementedError


class Z3CNN(CNN):
    def conv_bn_act(self, tensor, i):
        return layer.conv3d_bn_act(tensor, nb_channels_out=self.filters[i])


class GCNN(CNN):
    def conv_bn_act(self, tensor, i):
        in_group = 'Z3' if i == 0 else 'O'
        out_group = 'O'
        in_channels = self.filters[i - 1] if i != 0 else None
        return gconv.gconv3d_bn_act(tensor, in_group=in_group, out_group=out_group,
                                    in_channels=in_channels, out_channels=self.filters[i])


class MultiDim(BaseModel):
    def build_graph(self):
        self.filters = [16, 32, 64]
        tensor = self.conv3d_bn_act(self.x, 0)
        tensor = base.maxpool3d(tensor, strides=[1, 2, 2, 2, 1])

        tensor = self.block_2d(tensor)

        tensor = self.conv3d_bn_act(tensor, 1)
        tensor = base.maxpool3d(tensor, strides=[1, 2, 2, 2, 1])

        tensor = self.block_2d(tensor)

        tensor = self.conv3d_bn_act(tensor, 2)

        tensor = base.flatten(tensor)
        final = base.readout(tensor, [int(tensor.get_shape()[-1]), self.data.nb_classes])
        return final

    def block_2d(self, tensor):
        tensor, z, c = base.dim_reshape(tensor)
        tensor = self.conv2d_bn_act(tensor, out_channels=c)
        tensor = self.conv2d_bn_act(tensor, out_channels=c)
        tensor = base.dim_reshape(tensor, z)
        return tensor

    def conv3d_bn_act(self, tensor, i):
        raise NotImplementedError

    def conv2d_bn_act(self, tensor, out_channels):
        raise NotImplementedError


class Z3MultiDim(MultiDim):
    def conv3d_bn_act(self, tensor, i):
        return layer.conv3d_bn_act(tensor, [3, 3, 3], nb_channels_out=self.filters[i])

    def conv2d_bn_act(self, tensor, out_channels):
        return layer.conv2d_bn_act(tensor, nb_channels_out=out_channels)


class GMultiDim(MultiDim):
    def conv3d_bn_act(self, tensor, i):
        in_group = 'Z3' if i == 0 else 'O'
        out_group = 'O'
        in_channels = self.filters[i - 1] if i != 0 else None
        return gconv.gconv3d_bn_act(tensor, in_group=in_group, out_group=out_group,
                                    in_channels=in_channels, out_channels=self.filters[i])

    #
    # def conv2d_bn_act(self, tensor, out_channels):
    #     return gconv.gconv_bn_act(tensor, in_group='C4', out_group='C4', out_channels=out_channels)
    def conv2d_bn_act(self, tensor, out_channels):
        return layer.conv2d_bn_act(tensor, nb_channels_out=out_channels)


class Resnet(BaseModel):
    def build_graph(self):
        # self.filters = [16, 32, 64, 128]

        self.filters = [16, 16, 32, 32]

        blocks = len(self.filters) - 1

        # channels = 16
        tensor = self.conv_bn_act(self.x, None, self.filters[0], first=True)
        tensor = self.residual_block(tensor, self.filters[0])

        # channels = 32, 64
        for i in xrange(1, blocks):
            in_channels = self.filters[i - 1]
            out_channels = self.filters[i]

            tensor = self.conv_bn_act(tensor, in_channels, out_channels)
            tensor = self.residual_block(tensor, out_channels)
            tensor = base.maxpool3d(tensor, strides=[1, 2, 2, 2, 1])

        # channels = 128
        tensor = self.conv_bn_act(tensor, self.filters[-2], self.filters[-1])

        tensor = base.flatten(tensor)
        final = base.readout(tensor, [int(tensor.get_shape()[-1]), self.data.nb_classes])
        return final

    def residual_block(self, input_tensor, out_channels):
        tensor = self.conv_bn_act(input_tensor, out_channels, out_channels)
        tensor = self.conv_bn_act(tensor, out_channels, out_channels)
        tensor = base.merge(input_tensor, tensor, method='add')

        return tensor

    def conv_bn_act(self, tensor, in_channels, out_channels, first=False):
        raise NotImplementedError


class Z3Resnet(Resnet):
    def conv_bn_act(self, tensor, in_channels, out_channels, first=False):
        return layer.conv3d_bn_act(tensor, nb_channels_out=out_channels)


class GResnet(Resnet):
    def conv_bn_act(self, tensor, in_channels, out_channels, first=False):
        in_group = 'Z3' if first else 'O'
        return gconv.gconv3d_bn_act(tensor, in_group=in_group, out_group='O',
                                    in_channels=in_channels, out_channels=out_channels)
