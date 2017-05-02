import random
import copy
import numpy as np
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.matrix_garray import MatrixGArray
from groupy.garray.Oht_array import OhtArray
from groupy.garray.Z3_array import Z3Array


class OhArray(MatrixGArray):
    parameterizations = ['int', 'mat', 'hmat']
    _g_shapes = {'int': (2,), 'mat': (3, 3), 'hmat': (4, 4)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'Oh'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        self._left_actions[OhArray] = self.__class__.left_action_hmat
        self._left_actions[OhtArray] = self.__class__.left_action_hmat
        self._left_actions[Z3Array] = self.__class__.left_action_vec
        super(OhArray, self).__init__(data, p)
        self.elements = self.get_elements()

    def mat2int(self, mat_data):
        input = mat_data.reshape((-1, 3, 3))
        data = np.zeros((input.shape[0], 2), dtype=np.int)
        for i in xrange(input.shape[0]):
            mat = input[i]
            index, mirror = self.get_int(mat)
            data[i, 0] = index
            data[i, 1] = mirror
        data = data.reshape(mat_data.shape[:-2] + (2,))
        return data

    def get_int(self, mat_data):
        assert mat_data.shape == (3, 3)
        orig_data = copy.deepcopy(mat_data)
        m = 0 if orig_data.tolist() in self.elements else 1
        orig_data[0:3] = orig_data[0:3] * ((-1) ** m)
        i = self.elements.index(orig_data.tolist())
        return i, m

    def int2mat(self, int_data):
        index = int_data[..., 0].flatten()
        m = int_data[..., 1].flatten()
        data = np.zeros((len(index),) + (3, 3), dtype=np.int)

        for j in xrange(len(index)):
            hmat = self.get_mat(index[j], m[j])
            data[j, 0:3, 0:3] = hmat

        data = data.reshape(int_data.shape[:-1] + (3, 3))
        return data

    def get_mat(self, index, mirror):
        element = copy.deepcopy(self.elements[index])
        element = np.array(element, dtype=np.int)
        element[0:3] = element[0:3] * ((-1) ** mirror)
        element = element.astype(dtype=np.int)
        return element

    def get_elements(self):
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

class OhGroup(FiniteGroup, OhArray):
    def __init__(self):
        OhArray.__init__(
            self,
            data=np.array([[i, j] for i in xrange(24) for j in xrange(2)]),
            p='int'
        )
        FiniteGroup.__init__(self, OhArray)

    def factory(self, *args, **kwargs):
        return OhArray(*args, **kwargs)

Oh = OhGroup()

def rand(size=()):
    data = np.zeros(size + (2,), dtype=np.int)
    data[..., 0] = np.random.randint(0, 24, size)
    data[..., 1] = np.random.randint(0, 2, size)
    return OhArray(data=data, p='int')

def identity(p='int'):
    # alternatively: last element of self.elements -> int: array([23, 0])
    li = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    e = OhArray(data=np.array(li, dtype=np.int), p='mat')
    return e.reparameterize(p)


def i_range():
    start = 0
    stop = 24
    m = np.zeros((stop - start, 2), dtype=np.int)
    m[:, 0] = np.arange(start, stop)
    return OhArray(m)


def m_range():
    start = 0
    stop = 2
    m = np.zeros((stop - start, 2), dtype=np.int)
    m[:, 1] = np.arange(start, stop)
    return OhArray(m)


def meshgrid(i=24, m=2):
    return OhArray([[[k, l] for l in xrange(m)] for k in xrange(i)])
