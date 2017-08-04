# file to set environment variables for working remotely with PyCharm IDE
import os
os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ['PATH']
os.environ['PATH'] = '/home/marysia/bin:/home/marysia/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/lib/jvm/java-8-oracle/bin:/usr/lib/jvm/java-8-oracle/db/bin:/usr/lib/jvm/java-8-oracle/jre/bin'
os.environ['CUDA_PATH'] = '/usr/local/cuda'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'
#os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64/'
#if 'PYTHONPATH' in os.environ.keys():
#    os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':/usr/local/cuda/lib64'