import model
from training import Train

# get model
network = model.CNN_3D()
model = network.get_model()

# train 
train = Train(model)
train.compile_model()
train.get_summary()



