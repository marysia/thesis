import random
import copy
import numpy as np
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.matrix_garray import MatrixGArray


class OhArray(MatrixGArray):
    parameterizations = ['int', 'hmat']
    _g_shapes = {'int': (2,), 'hmat': (4, 4)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'Oh'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        self._left_actions[OhArray] = self.__class__.left_action_hmat
        super(OhArray, self).__init__(data, p)
        self.elements = self.get_elements()

    def hmat2int(self, hmat_data):
        input = hmat_data.reshape((-1, 4, 4))
        data = np.zeros((input.shape[0], 2), dtype=np.int)
        for i in xrange(input.shape[0]):
            hmat = input[i]
            index, mirror = self.get_int(hmat)
            data[i, 0] = index
            data[i, 1] = mirror
        data = data.reshape(hmat_data.shape[:-2] + (2,))
        return data

    def get_int(self, hmat_data):
        assert hmat_data.shape == (4, 4)
        orig_data = copy.deepcopy(hmat_data)
        m = 0 if orig_data.tolist() in self.elements else 1
        orig_data[0:3] = orig_data[0:3] * ((-1) ** m)
        i = self.elements.index(orig_data.tolist())
        return i, m

    def int2hmat(self, int_data):
        index = int_data[..., 0].flatten()
        m = int_data[..., 1].flatten()
        data = np.zeros((len(index),) + (4, 4), dtype=np.int)

        for j in xrange(len(index)):
            hmat = self.get_hmat(index[j], m[j])
            data[j, 0:4, 0:4] = hmat

        data = data.reshape(int_data.shape[:-1] + (4, 4))
        return data
    def get_hmat(self, index, mirror):
        element = copy.deepcopy(self.elements[index])
        element = np.array(element, dtype=np.int)
        element[0:3] = element[0:3] * ((-1) ** mirror)
        element = element.astype(dtype=np.int)
        return element

    def get_elements(self):
        g1 = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]  # 90o degree rotation over x
        g2 = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]  # 90o degree rotation over y

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

# generators & special elements
g1 = OhArray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], p='hmat')  # 90o degree rotation over x
g2 = OhArray([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], p='hmat') # 90o degree rotation over y
g3 = OhArray([[-1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], p='hmat') # 90s degree rotation + mirror

def rand(size=()):
    data = np.zeros(size + (2,), dtype=np.int)
    data[..., 0] = np.random.randint(0, 24, size)
    data[..., 1] = np.random.randint(0, 2, size)
    return OhArray(data=data, p='int')

def identity(p='int'):
    # alternatively: last element of self.elements -> int: array([23, 0])
    li = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    e = OhArray(data=np.array(li, dtype=np.int), p='hmat')
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
