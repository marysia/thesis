import test_garray
#from groupy.garray import Ot_array
print 'Starting'

# a = Ot_array.rand(minu=-1, maxu=2, minv=-1, maxv=2, minw=-1, maxw=2, size=(2, 3))
# b = a.int2hmat(a.data)
# a.hmat2int(b)

test_garray.test_o_array()
test_garray.test_oh_array()
test_garray.test_ot_array()
test_garray.test_oht_array()
print 'Done'
