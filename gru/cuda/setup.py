from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gru_cuda',
    ext_modules=[
        CUDAExtension('gru_cuda', [
            'gru_cuda.cpp',
            'gru_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
