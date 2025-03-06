from glob import glob
import os
import platform
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, build_ext, naive_recompile

version = "0.1.4"

# Optional multithreaded build
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

ParallelCompile("NPY_NUM_BUILD_JOBS", needs_recompile=naive_recompile).install()

import os
conda = print(os.environ["CONDA_PREFIX"])

extra_compile_args=["-g", f"-I{conda}/include/eigen3/"]

# extra_link_args = None
extra_link_args = ["-g", f"-I{conda}/include/eigen3/"]
if platform.system() == "Darwin":
  # Compiling on macOS requires an installation of the Xcode Command Line Tools
  os.environ["CC"] = "g++"
  os.environ["CXX"] = "g++"
  # extra_link_args.extend(["-undefined", "dynamic_lookup"])

#include/eigen3

ext_modules = [
    Pybind11Extension(
        "ptdalgorithms.ptdalgorithmscpp_pybind",
        sorted(glob("src/*/*.cpp") + glob("src/*/*.c")),
#        cxx_std=11,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    ),
]

setup(name='ptdalgorithms',
      package_dir = {'ptdalgorithms': 'ptdalgorithms'},
    #   test_suite='tests',
      version=version,
      description='',
      long_description='',
      author='',
      author_email='',
      url='',
      packages = find_packages(),
    #   package_dir = {'SAP': 'SAP'},
    #   include_package_data = True,     
      extras_require={"test": "pytest"},
      cmdclass={"build_ext": build_ext}, 
      ext_modules=ext_modules)
