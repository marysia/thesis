import random
import numpy as np
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.matrix_garray import MatrixGArray

class OtArray(MatrixGArray):
    '''
    Implementation of space group Ot.
    '''
    parameterizations = ['int', 'hmat']
    _g_shapes = {'int': (4,), 'hmat': (4, 4)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'Ot'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        self._left_actions[OtArray] = self.__class__.left_action_hmat
        super(OtArray, self).__init__(data, p)
        self.base_elements = self.get_base_elements()

    def hmat2int(self, hmat_data):
        mat = hmat_data[0:3, 0:3]
        i = self.base_elements.index(mat.tolist())
        u, v, w, _ = hmat_data[:, 3]
        return (i, u, v, w)

    def int2hmat(self, int_data):
        i = int_data[..., 0]
        u = int_data[..., 1]
        v = int_data[..., 2]
        w = int_data[..., 3]
        data = np.zeros(int_data.shape[:-1] + (4, 4), dtype=np.int)
        if i.shape == ():
            mat = self.elements[i]
            data[..., 0:3, 0:3] = mat
            data[..., 3, 0] = u
            data[..., 3, 1] = v
            data[..., 3, 2] = w
            data[..., 3, 3] = 1
        elif i.shape == (1,):
            mat = self.elements[i[0]]
            data[..., 0:3, 0:3] = mat
            data[..., 3, 0] = u[0]
            data[..., 3, 1] = v[0]
            data[..., 3, 2] = w[0]
            data[..., 3, 3] = 1
        else:
            for j in xrange(int_data.shape[0]):
                for k in xrange(int_data.shape[1]):
                    mat = self.elements[i[j, k]]
                    data[j, k, 0:3, 0:3] = mat
                    data[j, k, 3, 0] = u[j, k]
                    data[j, k, 3, 1] = v[j, k]
                    data[j, k, 3, 2] = w[j, k]
                    data[j, k, 3, 3] = 1
        return data


    def get_base_elements(self):
        g1 = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # 90o degree rotation over x
        g2 = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # 90o degree rotation over y
        element_list = [g1, g2]
        current = g1
        while len(element_list) < 24:
            multiplier = random.choice(element_list)
            current = np.dot(np.array(current), np.array(multiplier)).tolist()
            if current not in element_list:
                element_list.append(current)
        element_list.sort()
        return element_list

def identity(shape=(), p='int'):
    li = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    e = OtArray(data=np.array(li, dtype=np.int), p='hmat')
    return e.reparameterize(p)


def meshgrid(minu, maxu, minv, maxv, minw, maxw):
    li = [[i, u, v, w] for i in xrange(24) for u in xrange(minu, maxu) for v in xrange(minv, maxv) for
     w in xrange(minw, maxw)]
    return OtArray(li, p='int')


