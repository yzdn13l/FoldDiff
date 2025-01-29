from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer_3D',
    ext_modules=[
        CUDAExtension('chamfer_3D', [
            "/".join(__file__.split('/')[:-1] + ['chamfer_cuda.cpp']),
            "/".join(__file__.split('/')[:-1] + ['chamfer3D.cu']),
        ]),
    ],
    extra_compile_args={'cxx': [], 'nvcc': ['-DTHRUST_IGNORE_CUB_VERSION_CHECK']},
    extra_cuda_cflags=['--compiler-bindir=/usr/bin/gcc-8'],
    cmdclass={
        'build_ext': BuildExtension
    })