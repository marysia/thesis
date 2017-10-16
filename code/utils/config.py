import os
HOME = os.getenv("HOME")

vm = 'azure' if 'marysia' in HOME else 'aws'

DATADIR = os.path.join(HOME, 'data', 'thesis') if vm == 'azure' else '/mnt/thesis'
LOGDIR = os.path.join(HOME, 'thesis', 'logs')
RESULTSDIR = os.path.join(HOME, 'thesis', 'results')
