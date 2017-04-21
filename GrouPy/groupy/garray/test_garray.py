
# TODO: reshaping / flattening tests, check updating of shape, g_shape, ndim, g_ndim
# TODO: test all left_actions, not just composition in group


def test_p4_array():
    from groupy.garray import p4_array
    check_wallpaper_group(p4_array, p4_array.P4Array)


def test_p4m_array():
    from groupy.garray import p4m_array
    check_wallpaper_group(p4m_array, p4m_array.P4MArray)


def test_z2_array():
    from groupy.garray import Z2_array
    check_wallpaper_group(Z2_array, Z2_array.Z2Array)


def test_c4_array():
    from groupy.garray import C4_array
    check_finite_group(C4_array, C4_array.C4Array, C4_array.C4)


def test_d4_array():
    from groupy.garray import D4_array
    check_finite_group(D4_array, D4_array.D4Array, D4_array.D4)

def test_o_array():
    from groupy.garray import O_array
    check_o_group(O_array, O_array.OArray, O_array.O)

def test_oh_array():
    from groupy.garray import Oh_array
    check_o_group(Oh_array, Oh_array.OhArray, Oh_array.Oh)


def check_o_group(garray_module, garray_class, G):
    a = garray_module.rand()
    b = garray_module.rand()
    c = garray_module.rand()

    check_associative(a, b, c)
    check_identity(garray_module, a)
    check_inverse(garray_module, a)
    check_closed_under_composition_tmp(G)
    check_closed_under_inverses_tmp(G)
    check_reparameterize_invertible(garray_class, a)

def check_space_group(garray_module, garray_class):
    a = garray_module.rand(minu=-1, maxu=2, minv=-1, maxv=2, minw=1, maxw=2, size=(2, 3))
    b = garray_module.rand(minu=-1, maxu=2, minv=-1, maxv=2, minw=1, maxw=2, size=(2, 3))
    c = garray_module.rand(minu=-1, maxu=2, minv=-1, maxv=2, minw=1, maxw=2, size=(2, 3))

    check_associative(a, b, c)
    check_identity(garray_module, a)
    check_inverse(garray_module, a)

    check_reparameterize_invertible(garray_class, a)

    m = garray_module.meshgrid(minu=-1, maxu=2, minv=-1, maxv=2, minw=-1, maxw=2)
    check_closed_inverse(m)

def check_wallpaper_group(garray_module, garray_class):

    a = garray_module.rand(minu=-1, maxu=2, minv=-1, maxv=2, size=(2, 3))
    b = garray_module.rand(minu=-1, maxu=2, minv=-1, maxv=2, size=(2, 3))
    c = garray_module.rand(minu=-1, maxu=2, minv=-1, maxv=2, size=(2, 3))

    check_associative(a, b, c)
    check_identity(garray_module, a)
    check_inverse(garray_module, a)

    check_reparameterize_invertible(garray_class, a)

    m = garray_module.meshgrid(
        u=garray_module.u_range(-1, 2),
        v=garray_module.v_range(-1, 2)
    )
    check_closed_inverse(m)


def check_finite_group(garray_module, garray_class, G):

    a = garray_module.rand()
    b = garray_module.rand()
    c = garray_module.rand()

    check_associative(a, b, c)
    check_identity(garray_module, a)
    check_inverse(garray_module, a)

    check_reparameterize_invertible(garray_class, a)

    check_closed_composition(G)
    check_closed_inverse(G)


def check_associative(a, b, c):
    ab = a * b
    ab_c = ab * c
    bc = b * c
    a_bc = a * bc
    assert (ab_c == a_bc).all()


def check_identity(garray_module, a):
    e = garray_module.identity()
    assert (e * a == a).all()
    assert (a * e == a).all()


def check_inverse(garray_module, a):
    e = garray_module.identity()
    assert (a * a.inv() == e).all()
    assert (a.inv().inv() == a).all()



# TODO: could not use check_closed_composition and check_closed_inverse for O group
# Error for Gf[:, None]*Gf[None,:] and Gf.inv() respectively
# Traceback:
#   -> self_mat = self.reparameterize(mat_p).data
#   -> new_data = self._reparameterizations[(self.p, p)](self.data)
#   ->  return np.array(self._elements[int_data], dtype=np.int)
# TypeError: only integer arrays with one element can be converted to an index
# Temporary solution: redefine closed under composition and closed under inverses
def check_closed_under_composition_tmp(G):
    '''
    This function creates a set of products of each element combination
    in G. This set must equal the original G for G to be closed under composition.

    :param G: a GArray containing every element of a finite group.
    '''
    original_group = []
    product_group = []
    for a in G:
        original_group.append(a)
        for b in G:
            product = a * b
            assert product in G
            if product not in product_group:
                product_group.append(product)
    check_same_groups(original_group, product_group)

def check_closed_under_inverses_tmp(G):
    '''
    This function creates a set of inverses of each element of G.
    This set must equal the original G for G to be closed under inverses.
    '''
    original_group = []
    inverse_group = []
    for a in G:
        original_group.append(a)
        inverse = a.inv()
        assert a in G
        if inverse not in inverse_group:
            inverse_group.append(inverse)
    check_same_groups(original_group, inverse_group)

def check_same_groups(G, H):
    '''
    Groups G and H are the same if they are the same size
    and every element in G is in H, and vice versa.
    '''
    assert len(G) == len(H)
    for a in G:
        assert a in H
    for b in H:
        assert b in G

def check_garray_equal_as_sets(G, H):
    """
    Check that two GArrays G and H are equal as sets,
    i.e. that every element in G is in H and vice versa.
    """
    Gf = G.flatten()
    Hf = H.flatten()

    for i in range(Gf.size):
        gi = Gf[i]
        assert (gi == H).sum() > 0

    for i in range(Hf.size):
        hi = Hf[i]
        assert (hi == G).sum() > 0

def check_closed_composition(G):
    """
    Check that a finite group G is closed under the group operation.
    This function computes an "outer product" of the GArray G,
    i.e. each element of G is multiplied with each other element.
    Then, we check that the resulting elements are all in G,
    and that each row and column of the outer product is equal to G as a set.

    :param G: a GArray containing every element of a finite group.
    """

    Gf = G.flatten()
    outer = Gf[:, None] * Gf[None, :]

    for i in range(outer.shape[0]):
        Gi = outer[i, :]
        assert Gi.size == G.size
        check_garray_equal_as_sets(G, Gi)

        Gi = outer[:, i]
        assert Gi.size == G.size
        check_garray_equal_as_sets(G, Gi)


def check_closed_inverse(G):
    """
    Check that a finite group G is closed under the inverses.
    This function computes the inverse of each element in G,
    and then checks that the resulting set is equal to G as a set.

    Note: this function can be used on finite groups G,
    but also on "symmetric sets" in infinite groups.
    I define a symmetric set as a subset of a group that is closed under inverses,
    but not necessarily under composition.
    An example are the translations by up to and including 1 unit in x and y direction,
    composed with every rotation in the group p4.

    :param G: a GArray containing every element of a finite group.
    """

    Gf = G.flatten()
    Ginv = Gf.inv()
    check_garray_equal_as_sets(G, Ginv)


def check_reparameterize_invertible(garray_class, a):
    import copy

    for p1 in garray_class.parameterizations:

        b = copy.deepcopy(a)
        bp1 = b.reparameterize(p1)
        bp1data = bp1.data.copy()

        for p2 in garray_class.parameterizations:
            bp2 = bp1.reparameterize(p2)
            bp21 = bp2.reparameterize(p1)
            assert (bp1data == bp21.data).all()
