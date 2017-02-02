import model
from training import Train
from keras.optimizers import SGD
network = model.Resnet()
train = Train(network.model)
train.compile_model()
train.get_summary()
