import random
import numpy as np
import copy
from groupy.garray.finitegroup import FiniteGroup
from groupy.garray.matrix_garray import MatrixGArray
from groupy.garray.Ot_array import OtArray
from groupy.garray.Z3_array import Z3Array


class OArray(MatrixGArray):
    parameterizations = ['int', 'mat', 'hmat']
    _g_shapes = {'int': (1,), 'mat': (3, 3), 'hmat': (4, 4)}
    _left_actions = {}
    _reparameterizations = {}
    _group_name = 'O'

    def __init__(self, data, p='int'):
        data = np.asarray(data)
        assert data.dtype == np.int
        self._left_actions[OArray] = self.__class__.left_action_hmat
        self._left_actions[OtArray] = self.__class__.left_action_hmat
        self._left_actions[Z3Array] = self.__class__.left_action_vec

        super(OArray, self).__init__(data, p)
        self.elements = self.get_elements()

    def mat2int(self, hmat_data):
        input = hmat_data.reshape((-1, 3, 3))
        data = np.zeros((input.shape[0], 1), dtype=np.int)
        for i in xrange(input.shape[0]):
            mat = input[i]
            index = self.elements.index(mat.tolist())
            data[i, 0] = index
        data = data.reshape(hmat_data.shape[:-2] + (1,))
        return data

    def int2mat(self, int_data):
        i = int_data[..., 0].flatten()
        data = np.zeros((len(i),) + (3, 3), dtype=np.int)

        for j in xrange(len(i)):
            mat = self.elements[i[j]]
            data[j, 0:3, 0:3] = mat

        data = data.reshape(int_data.shape[:-1] + (3, 3))
        return data



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

def rand(size=()):
    data = np.zeros(size + (1,), dtype=np.int64)
    data[..., 0] = np.random.randint(0, 24, size)
    return OArray(data=data, p='int')

def identity(p='int'):
    # alternatively: last element of self.elements
    li = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    e = OArray(data=np.array(li, dtype=np.int), p='mat')
    return e.reparameterize(p)
