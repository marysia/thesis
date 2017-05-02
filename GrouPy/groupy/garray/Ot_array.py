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
        out = np.zeros(hmat_data.shape[:-2] + (4,), dtype=np.int)

        # handle different input shapes
        if len(hmat_data.shape) == 2:
            mat = hmat_data[0:3, 0:3]
            u, v, w, _ = hmat_data[:, 3]
            index = self.base_elements.index(mat.tolist())
            out[..., 0] = index
            out[..., 1] = u
            out[..., 2] = v
            out[..., 3] = w
        elif len(hmat_data.shape) == 3:
            for j in xrange(hmat_data.shape[0]):
                mat = hmat_data[j][0:3, 0:3]
                u, v, w, _ = hmat_data[j][:, 3]
                index = self.base_elements.index(mat.tolist())
                out[j, 0] = index
                out[j, 1] = u
                out[j, 2] = v
                out[j, 3] = w
        else:
            for i in xrange(hmat_data.shape[0]):
                for j in xrange(hmat_data.shape[1]):
                    mat = hmat_data[i, j][0:3, 0:3]
                    u, v, w, _ = hmat_data[i, j][:, 3]
                    index = self.base_elements.index(mat.tolist())
                    out[i, j, 0] = index
                    out[i, j, 1] = u
                    out[i, j, 2] = v
                    out[i, j, 3] = w
        return out

    def int2hmat(self, int_data):
        i = int_data[..., 0]
        u = int_data[..., 1]
        v = int_data[..., 2]
        w = int_data[..., 3]
        data = np.zeros(int_data.shape[:-1] + (4, 4), dtype=np.int)
        if i.shape == ():
            mat = self.base_elements[i]
            data[..., 0:3, 0:3] = mat
            data[..., 0, 3] = u
            data[..., 1, 3] = v
            data[..., 2, 3] = w
            data[..., 3, 3] = 1
        elif len(i.shape) == 1:
            for j in xrange(i.shape[0]):
                #TODO: THIS IS NOT YET UPDATED ON GPU01
                mat = self.base_elements[i[j]]
                data[j, 0:3, 0:3] = mat
                data[j, 0, 3] = u[j]
                data[j, 1, 3] = v[j]
                data[j, 2, 3] = w[j]
                data[j, 3, 3] = 1
        if len(i.shape) == 2:
            for j in xrange(int_data.shape[0]):
                for k in xrange(int_data.shape[1]):
                    mat = self.base_elements[i[j, k]]
                    data[j, k, 0:3, 0:3] = mat
                    data[j, k, 0, 3] = u[j, k]
                    data[j, k, 1, 3] = v[j, k]
                    data[j, k, 2, 3] = w[j, k]
                    data[j, k, 3, 3] = 1
        if len(i.shape) == 4:
            for j in xrange(int_data.shape[0]):
                for k in xrange(int_data.shape[1]):
                    for l in xrange(int_data.shape[2]):
                        for m in xrange(int_data.shape[3]):
                            mat = self.base_elements[i[j, k, l, m]]
                            data[j, k, l, m, 0:3, 0:3] = mat
                            data[j, k, l, m, 0, 3] = u[j, k, l, m]
                            data[j, k, l, m, 1, 3] = v[j, k, l, m]
                            data[j, k, l, m, 2, 3] = w[j, k, l, m]
                            data[j, k, l, m, 3, 3] = 1

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

def rand(minu, maxu, minv, maxv, minw, maxw, size=()):
    data = np.zeros(size + (4, ), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 24, size)
    data[..., 1] = np.random.randint(minu, maxu, size)
    data[..., 2] = np.random.randint(minv, maxv, size)
    data[..., 3] = np.random.randint(minw, maxw, size)
    return OtArray(data=data, p='int')

def meshgrid(minu=-1, maxu=2, minv=-1, maxv=2, minw=-1, maxw=2):

    li = [[i, u, v, w] for i in xrange(24) for u in xrange(minu, maxu) for v in xrange(minv, maxv) for
     w in xrange(minw, maxw)]
    return OtArray(li, p='int')


