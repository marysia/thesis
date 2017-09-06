from groupy.gconv.chainer_gconv.splitgconv3d import SplitGConv3D
from groupy.gconv.make_gconv_indices import make_o_z3_indices, make_o_ot_indices


class OtConvZ3(SplitGConv3D):
    input_stabilizer_size = 1
    output_stabilizer_size = 24

    def make_transformation_indices(self, ksize):
        return make_o_z3_indices(ksize=ksize)


class OtConvOt(SplitGConv3D):
    input_stabilizer_size = 24
    output_stabilizer_size = 24

    def make_transformation_indices(self, ksize):
        return make_o_ot_indices(ksize=ksize)
