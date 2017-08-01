import utils.environment
from preprocessing.patches import DataPatches
from preprocessing.generic import data_metrics
from models.model_3d import Z3CNN, GCNN, Resnet, GResnet
import sys
from utils.control import ProgramEnder
from utils.logger import Logger
from models.util.helpers import total_parameters

models = {
#    'Z3CNN': Z3CNN,
    'GCNN': GCNN
#    'Resnet': Resnet,
#    'GResnet': GResnet
}
log = Logger('/home/marysia/thesis/logs/')
#log.backup_additional(['models/model_3d.py'])

log.info('Executing patches script with the following models: %s' % str(models.keys()))
ender = ProgramEnder()
data = DataPatches(small=True)
data_metrics(data, log)
tot_params = 0

for name, model in models.items():
    if not ender.terminate:
        log.info('\n\nStarting model ' + name)
        # initialize
        graph = model(model_name=name, data=data, epochs=1, ender=ender, log=log, verbose=False)
        # build
        graph.build_model()
        model_params = total_parameters() - tot_params
        tot_params += model_params

        log.info('Model: Number of parameters is... ' + str(model_params))
        # train
        #graph.train(mode='time', mode_param=2, save_step=4)
        graph.train(mode='epochs', mode_param=1, save_step=15)
    else:
        log.info('Program was terminated. Model %s will not be trained and/or evaluated.' % name)


log.info('Executed all steps in script.\n')
