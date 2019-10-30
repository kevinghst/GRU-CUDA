from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='gru_tanh_cpp',
    ext_modules=[
        CppExtension('gru_tanh_cpp', ['gru.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })