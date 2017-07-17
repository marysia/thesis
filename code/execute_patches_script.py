import utils.environment
from preprocessing.patches import DataPatches
from preprocessing.generic import data_metrics
from models.model_3d import Z3CNN, GCNN, Resnet

from utils.control import ProgramEnder
from models.util.helpers import total_parameters

models = {
   #  'Z3CNN': Z3CNN,
   #  'GCNN': GCNN,
    'Resnet': Resnet
}

ender = ProgramEnder()
data = DataPatches()
data_metrics(data)
tot_params = 0

for name, model in models.items():
    if not ender.terminate:
        print('\nStarting model ' + name)
        # initialize
        graph = model(model_name=name, data=data, epochs=15, ender=ender, verbose=False)
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
