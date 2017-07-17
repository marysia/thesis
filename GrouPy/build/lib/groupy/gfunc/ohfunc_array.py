import groupy.garray.Oh_array as oh
from groupy.gfunc.gfuncarray import GFuncArray


class OhFuncArray(GFuncArray):
    def __init__(self, v):
        i2g = oh.meshgrid()
        super(OhFuncArray, self).__init__(v=v, i2g=i2g)

    def g2i(self, g):
        gint = g.reparameterize('int').data.copy()
        return gint