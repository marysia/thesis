import random
import numpy as np
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.matrix_garray import MatrixGArray


class OhArray(MatrixGArray):
    parameterizations = ['int', 'hmat']
    _g_shapes = {'int': (1,), 'hmat': (4, 4)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'O'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        self._left_actions[OhArray] = self.__class__.left_action_hmat
        super(OhArray, self).__init__(data, p)
        self.elements = self.get_elements()

    def hmat2int(self, mat_data):
        return (self.elements.index(mat_data.tolist()), )
    def int2hmat(self, int_data):
        return np.array(self.elements[int_data], dtype=np.int)
    def get_elements(self):
        g1 = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]  # 90o degree rotation over x
        g2 = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]  # 90o degree rotation over y
        g3 = [[-1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]] # 90o degree rotation + mirror
        element_list = [g1, g2, g3]
        current = g1
        while len(element_list) < 48:
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
            data=np.arange(48)[:, None],
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
    data = np.zeros(size + (1,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 48, size)
    return OhArray(data=data, p='int')

def identity(p='int'):
    # alternatively: last element of self._elements
    li = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    e = OhArray(data=np.array(li, dtype=np.int), p='hmat')
    return e.reparameterize(p)
