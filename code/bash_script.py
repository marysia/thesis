import os
import time

def froc_curves():
    cmd = "python noduleCADEvaluationLUNA16.py --final --constraints 1000 --fname --fname froc-1000"
    os.system(cmd)
    cmd = "python noduleCADEvaluationLUNA16.py --final --constraints 5000 --fname --fname froc-5000"
    os.system(cmd)
    cmd = "python noduleCADEvaluationLUNA16.py --final --constraints 10000 --fname --fname froc-10000"
    os.system(cmd)
    cmd = "python noduleCADEvaluationLUNA16.py --final --constraints 15000 --fname --fname froc-15000"
    os.system(cmd)
    cmd = "python noduleCADEvaluationLUNA16.py --final --constraints z3 --fname froc-z3"
    os.system(cmd)
    cmd = "python noduleCADEvaluationLUNA16.py --final --constraints c4h --fname froc-c4h"
    os.system(cmd)
    cmd = "python noduleCADEvaluationLUNA16.py --final --constraints d4h --fname froc-d4h"
    os.system(cmd)
    cmd = "python noduleCADEvaluationLUNA16.py --final --fname froc-all"
    os.system(cmd)


def train_runs():
    cuda = "CUDA_VISIBLE_DEVICES=1 "

    cmd = cuda + "python train_script.py --log 1000 --groups z3 c4h d4h o --mode time --mode_param 300 --samples 1000 --shape 12 72 72 --augment scale flip rotate noise --symmetry --submission"
    os.system(cmd)

    cmd = cuda + "python train_script.py --log 5000 --groups z3 c4h d4h o --mode time --mode_param 300 --samples 5000 --shape 12 72 72 --augment scale flip rotate noise --symmetry --submission"
    os.system(cmd)

    cmd = cuda + "python train_script.py --log 10000 --groups z3 c4h d4h o --mode time --mode_param 300 --samples 10000 --shape 12 72 72 --augment scale flip rotate noise --symmetry --submission"
    os.system(cmd)

    cmd = cuda + "python train_script.py --log 15000 --groups z3 c4h d4h o --mode time --mode_param 300 --samples 15000 --shape 12 72 72 --augment scale flip rotate noise --symmetry --submission"
    os.system(cmd)


# cmd1 = "CUDA_VISIBLE_DEVICES=0 python train_script.py --log cmd1 --groups z3 --mode 3 --mode_param 3 --samples 000 --shape 12 72 72 --augment flip rotate noise --symmetry"
# cmd2 = "CUDA_VISIBLE_DEVICES=1 python train_script.py --log cmd2 --groups b --mode 3 --mode_param 3 --samples 5000 --shape 12 72 72 --augment flip rotate noise --symmetry"
# os.system(cmd1)
# os.system(cmd2)

train_runs()