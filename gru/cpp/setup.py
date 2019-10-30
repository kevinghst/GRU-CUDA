from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='gru_cpp',
    ext_modules=[
        CppExtension('gru_cpp', ['gru.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })