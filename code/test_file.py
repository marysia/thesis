import model
from training import Train
from keras.optimizers import SGD
network = model.Resnet3D(input_shape=(1, 7, 72, 72))
print network.input_shape
print network.order
print network.blocks
print network.name

train = Train(network.model)
train.compile_model()
train.get_summary()