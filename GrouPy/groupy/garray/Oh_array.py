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
        out = np.zeros(hmat_data.shape[:-2] + (2,), dtype=np.int)
        if len(hmat_data.shape) != 4:
            index, mirror = self.get_int(hmat_data)
            out[..., 0] = index
            out[..., 1] = mirror
        else:
            for i in xrange(hmat_data.shape[0]):
                for j in xrange(hmat_data.shape[1]):
                    index, mirror = self.get_int(hmat_data[i, j])
                    out[i, j, 0] = index
                    out[i, j, 1] = mirror
        return out

    def get_int(self, hmat_data):
        orig_data = copy.deepcopy(hmat_data)
        m = 0 if orig_data.tolist() in self.elements else 1
        orig_data[0:3] = orig_data[0:3] * ((-1) ** m)
        i = self.elements.index(orig_data.tolist())
        return i, m

    def int2hmat(self, int_data):
        index = int_data[..., 0]
        m = int_data[..., 1]
        out = np.zeros(int_data.shape[:-1] + (4, 4), dtype=np.int)

        # quick fix.
        if index.shape == ():
            hmat = self.get_hmat(index, m)
            out[..., 0:4, 0:4] = hmat

            # for k in range(4):
            #     for l in range(4):
            #         out[..., k, l] = hmat[k, l]
        elif index.shape == (1,):
            hmat = self.get_hmat[index[0], m[0]]
            out[..., 0:4, 0:4] = hmat
            # for k in range(4):
            #     for l in range(4):
            #         out[..., k, l] = hmat[k, l]
        else:
            for i in xrange(index.shape[0]):
                for j in xrange(index.shape[1]):
                    hmat = self.get_hmat(index[i,j], m[i, j])
                    out[i, j, 0:4, 0:4] = hmat
                    # for k in range(4):
                    #     for l in range(4):
                    #         out[i, j, k, l] = hmat[k, l]

        return out

    def get_hmat(self, index, mirror):
        element = copy.deepcopy(self.elements[index])
        element = np.array(element, dtype=np.int)
        element[0:3] = element[0:3] * ((-1) ** mirror)
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
    data = np.zeros(size + (2,), dtype=np.int64)
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
