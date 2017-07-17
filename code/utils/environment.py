# file to set environment variables for working remotely with PyCharm IDE
import os
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ['PATH']
os.environ['CUDA_PATH'] = '/usr/local/cuda'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-8.0/lib64'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64/'
if 'PYTHONPATH' in os.environ.keys():
    os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':/usr/local/cuda/lib64'