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
        input = hmat_data.reshape((-1, 4, 4))
        data = np.zeros((input.shape[0], 4), dtype=np.int)
        for i in xrange(input.shape[0]):
            hmat = input[i]
            mat = [elem[0:3] for elem in hmat.tolist()][0:3]
            index = self.base_elements.index(mat)
            u, v, w, _ = hmat[:, 3]
            data[i, 0] = index
            data[i, 1] = u
            data[i, 2] = v
            data[i, 3] = w
        data = data.reshape(hmat_data.shape[:-2] + (4,))
        return data

    def int2hmat(self, int_data):
        i = int_data[..., 0].flatten()
        u = int_data[..., 1].flatten()
        v = int_data[..., 2].flatten()
        w = int_data[..., 3].flatten()
        data = np.zeros((len(i),) + (4, 4), dtype=np.int)

        for j in xrange(len(i)):
            mat = self.base_elements[i[j]]
            data[j, 0:3, 0:3] = mat
            data[j, 0, 3] = u[j]
            data[j, 1, 3] = v[j]
            data[j, 2, 3] = w[j]
            data[j, 3, 3] = 1

        data = data.reshape(int_data.shape[:-1] + (4, 4))
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


def i_range(start=0, stop=24, step=1):
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 0] = np.arange(start, stop, step)
    return OtArray(m)


def u_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 1] = np.arange(start, stop, step)
    return OtArray(m)


def v_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 2] = np.arange(start, stop, step)
    return OtArray(m)

def w_range(start=-1, stop=2, step=1):
    m = np.zeros((stop - start, 4), dtype=np.int)
    m[:, 3] = np.arange(start, stop, step)
    return OtArray(m)

def meshgrid_old(i=i_range(), u=u_range(), v=v_range(), w=w_range()):
    i = OtArray(i.data[:, None, None, None, ...], p=i.p)
    u = OtArray(u.data[None, :, None, None, ...], p=u.p)
    v = OtArray(v.data[None, None, :, None, ...], p=u.p)
    w = OtArray(w.data[None, None, None, :, ...], p=v.p)
    #return u * v * m * r
    return v * w * i * u


def meshgrid(minu=-1, maxu=2, minv=-1, maxv=2, minw=-1, maxw=2):
    li = [[i, u, v, w] for i in xrange(24) for u in xrange(minu, maxu) for v in xrange(minv, maxv) for
     w in xrange(minw, maxw)]
    return OtArray(li, p='int')


