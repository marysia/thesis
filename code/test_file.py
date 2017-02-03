import model
from training import Train
from keras.optimizers import SGD
network = model.ZuidhofRN(blocks=2)
print network.input_shape
print network.blocks
print network.name

train = Train(network.model)
train.compile_model()
train.get_summary()