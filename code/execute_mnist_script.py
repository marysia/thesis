# Script to run various models for mnist-rot

# environment import is not used, but sets some environment variables through os.environ
import utils.environment
from preprocessing.mnist import DataMNIST
from preprocessing.generic import data_metrics
from models.model_2d import ConvolutionalModel1, GConvModel1, Resnet, GResnet, Z2CNN, P4CNN, P4CNNDropout
from utils.control import ProgramEnder
from models.util.helpers import total_parameters

models = {
#    'Resnet1': Resnet#,
#    'GResnet1': GResnet
#    'Conv1': ConvolutionalModel1,
#    'Gconv1': GConvModel1,
     'Z2CNN': Z2CNN,
     'P4CNN': P4CNN,
     'P4CNNDropout': P4CNNDropout
}

ender = ProgramEnder()

data = DataMNIST()
data_metrics(data)
tot_params = 0

for name, model in models.items():
    if not ender.terminate:
        print('\nStarting model ' + name)
        # initialize
        graph = model(model_name=name, data=data, epochs=5, ender=ender, verbose=True)
        # build
        graph.build_model()
        model_params = total_parameters() - tot_params
        tot_params += model_params
        print('Model: Number of parameters is... ' + str(model_params))
        # train
        graph.train(mode='epochs')
    else:
        print('Program was terminated. Model %s will not be trained and/or evaluated.' % name)


print('Executed all steps in script.')