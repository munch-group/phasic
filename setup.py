from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, build_ext, naive_recompile

# Optional multithreaded build
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

ParallelCompile("NPY_NUM_BUILD_JOBS", needs_recompile=naive_recompile).install()


ext_modules = [
    Pybind11Extension(
        "ptdalgorithms.ptdalgorithmscpp_pybind",
        sorted(glob("src/*/*.cpp") + glob("src/*/*.c"))# + glob("src/*/*.h") + glob("api/*/*.h")),
    ),
]

setup(name='ptdalgorithms',
      package_dir = {'ptdalgorithms': 'ptdalgorithms'},
    #   test_suite='tests',
      version='0.1.0',
      description='',
      long_description='',
      author='',
      author_email='',
      url='',
      packages = find_packages(),
    #   package_dir = {'SAP': 'SAP'},
    #   include_package_data = True,     
      cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
