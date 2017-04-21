import random
import copy
import numpy as np
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.matrix_garray import MatrixGArray


class OhtArray(MatrixGArray):
    parameterizations = ['int', 'hmat']
    _g_shapes = {'int': (5,), 'hmat': (4, 4)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'Oht'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        self._left_actions[OhtArray] = self.__class__.left_action_hmat
        super(OhtArray, self).__init__(data, p)
        self.base_elements = self.get_base_elements()

    def hmat2int(self, hmat_data):
        out = np.zeros(hmat_data.shape[:-2] + (5,), dtype=np.int)
        if len(hmat_data.shape) != 4:
            index, mirror = self.get_int(hmat_data)
            u, v, w, _ = hmat_data[:, 3]
            out[..., 0] = index
            out[..., 1] = mirror
            out[..., 2] = u
            out[..., 3] = v
            out[..., 4] = w
            out[..., 5] = 1
        else:
            for i in xrange(hmat_data.shape[0]):
                for j in xrange(hmat_data.shape[1]):
                    index, mirror = self.get_int(hmat_data[i, j][0:3, 0:3])
                    u, v, w, _ = hmat_data[i, j][:, 3]
                    out[i, j, 0] = index
                    out[i, j, 1] = mirror
                    out[i, j, 2] = u
                    out[i, j, 3] = v
                    out[i, j, 4] = w
                    out[i, j, 5] = 1
        return out

    def int2hmat(self, int_data):
        '''
        Transforms int parameterization to hmat
        Some shape-magic if statements to ensure any
        shape can be handled.
        '''
        i = int_data[..., 0]
        m = int_data[..., 1]
        u = int_data[..., 2]
        v = int_data[..., 3]
        w = int_data[..., 4]
        data = np.zeros(int_data.shape[:-1] + (4, 4), dtype=np.int)
        if i.shape == ():
            mat = self.get_mat(i, m)
            data[..., 0:3, 0:3] = mat
            data[..., 3, 0] = u
            data[..., 3, 1] = v
            data[..., 3, 2] = w
            data[..., 3, 3] = 1
        elif i.shape == (1,):
            mat = self.get_mat(i[0], m[0])
            data[..., 0:3, 0:3] = mat
            data[..., 3, 0] = u[0]
            data[..., 3, 1] = v[0]
            data[..., 3, 2] = w[0]
            data[..., 3, 3] = 1
        else:
            for j in xrange(int_data.shape[0]):
                for k in xrange(int_data.shape[1]):
                    mat = self.get_mat(i[j, k], m[j, k])
                    data[j, k, 0:3, 0:3] = mat
                    data[j, k, 3, 0] = u[j, k]
                    data[j, k, 3, 1] = v[j, k]
                    data[j, k, 3, 2] = w[j, k]
                    data[j, k, 3, 3] = 1
        return data

    def get_mat(self, index, mirror):
        '''
        Return matrix representation of a given int parameterization (index, mirror)
        by determining looking up the mat by index and mirroring if necessary
        (note: deepcopy to avoid alterations to original self.base_elements)
        '''
        element = copy.deepcopy(self.base_elements[index])
        element = np.array(element, dtype=np.int)
        element = element * ((-1) ** mirror)
        return element

    def get_int(self, hmat_data):
        '''
        Return int (index, mirror) representation of given mat
        by mirroring if necessary to find the original mat and
        looking up the index in the list of base elements
        '''
        orig_data = copy.deepcopy(hmat_data)
        m = 0 if orig_data.tolist() in self.elements else 1
        orig_data = orig_data * ((-1) ** m)
        i = self.base_elements.index(orig_data.tolist())
        return i, m


    def get_base_elements(self):
        '''
        Generate all base elements of the group (no translations).
        '''
        g1 = [[1, 0, 0], [0, 0, -1], [0, 1, 0], [0, 0, 0]]  # 90o degree rotation over x
        g2 = [[0, 0, 1], [0, 1, 0], [-1, 0, 0], [0, 0, 0]]  # 90o degree rotation over y

        element_list = [g1, g2]
        current = g1
        while len(element_list) < 24:
            multiplier = random.choice(element_list)
            current = np.dot(np.array(current), np.array(multiplier)).tolist()
            if current not in element_list:
                element_list.append(current)
            element_list.sort()
        return element_list


def rand(minu, maxu, minv, maxv, minw, maxw, size=()):
    data = np.zeros(size + (5, ), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 24, size)
    data[..., 1] = np.random.randint(0, 2, size)
    data[..., 2] = np.random.randint(minu, maxu, size)
    data[..., 3] = np.random.randint(minv, maxv, size)
    data[..., 4] = np.random.randint(minw, maxw, size)
    return OhtArray(data=data, p='int')

def identity(p='int'):
    li = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    e = OhtArray(data=np.array(li, dtype=np.int), p='hmat')
    return e.reparameterize(p)

def meshgrid(minu, maxu, minv, maxv, minw, maxw):
    li = [[i, m, u, v, w] for i in xrange(24) for m in xrange(2) for u in xrange(minu, maxu) for v in xrange(minv, maxv) for
     w in xrange(minw, maxw)]
    return OhtArray(li, p='int')

