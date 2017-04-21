import random
import numpy as np
import copy
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.matrix_garray import MatrixGArray


class OArray(MatrixGArray):
    parameterizations = ['int', 'hmat']
    _g_shapes = {'int': (1,), 'hmat': (4, 4)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'O'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        self._left_actions[OArray] = self.__class__.left_action_hmat
        super(OArray, self).__init__(data, p)
        self.elements = self.get_elements()

    def hmat2int(self, mat_data):

        return (self.elements.index(mat_data.tolist()), )
    def int2hmat(self, int_data):
        i = int_data[..., 0]
        data = np.zeros(int_data.shape[:-1] + (4, 4), dtype=np.int)
        if i.shape == ():
            hmat = self.elements[i]
            data[..., 0:4, 0:4] = hmat
        elif i.shape == (1,):
            hmat = self.elements[i[0]]
            data[..., 0:4, 0:4] = hmat
        else:
            for j in xrange(int_data.shape[0]):
                for k in xrange(int_data.shape[1]):
                    hmat = self.elements[i[j, k]]
                    data[j, k, 0:4, 0:4] = hmat
        return data
        #return np.array(self.elements[int_data], dtype=np.int)


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

class OGroup(FiniteGroup, OArray):
    def __init__(self):
        OArray.__init__(
            self,
            data=np.arange(24)[:, None],
            p='int'
        )
        FiniteGroup.__init__(self, OArray)

    def factory(self, *args, **kwargs):
        return OArray(*args, **kwargs)

O = OGroup()

# generators & special elements
g1 = OArray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], p='hmat')  # 90o degree rotation over x
g2 = OArray([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], p='hmat') # 90o degree rotation over y

def rand(size=()):
    data = np.zeros(size + (1,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 24, size)
    return OArray(data=data, p='int')

def identity(p='int'):
    # alternatively: last element of self._elements
    li = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    e = OArray(data=np.array(li, dtype=np.int), p='hmat')
    return e.reparameterize(p)
