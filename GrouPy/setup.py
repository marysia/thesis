#!/usr/bin/env python

from distutils.core import setup

setup(
    name='GrouPy',
    version='0.1',
    description='Group equivariant convolutional neural networks',
    author='Taco S. Cohen',
    author_email='taco.cohen@gmail.com',
    packages=['groupy', 'groupy.garray', 'groupy.gconv', 'groupy.gfunc', 'groupy.gfunc.plot',
              'groupy.gconv.chainer_gconv', 'groupy.gconv.chainer_gconv.kernels', 'groupy.gconv.chainer_gconv.pooling',
              'groupy.gconv.tensorflow_gconv'],
)
