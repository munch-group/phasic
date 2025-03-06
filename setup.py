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
      ext_modules=ext_modules,
      # package_data={
      # 'ptdalgorithms.data.src.c': ['src/c/*'],
      # 'ptdalgorithms.data.src.cpp': ['src/cpp/*'],
      # 'ptdalgorithms.data.api.c': ['api/c/*'],
      # 'ptdalgorithms.data.api.cpp': ['api/cpp/*'],
      #  },
      data_files=[
        ('include', ['ptdalgorithms/ptdalgorithms.h']),
        ('include/ptdalgorithms/api/c', ['api/c/ptdalgorithms.h']),
        ('include/ptdalgorithms/api/cpp', ['api/cpp/ptdalgorithmscpp.h']),
        ('include/ptdalgorithms/src/c', ['src/c/ptdalgorithms.c', 'src/c/ptdalgorithms.h']),
        ('include/ptdalgorithms/src/cpp', ['src/cpp/ptdalgorithmscpp.cpp', 'src/cpp/ptdalgorithmscpp.h', 'src/cpp/ptdalgorithmscpp_pybind.cpp']),

        # ('include/ptdalgorithms/src/c', ['src/c/ptdalgorithms.c', 'src/c/ptdalgorithms.h']),
        # ('include/ptdalgorithms/src/cpp', ['src/cpp/ptdalgorithmscpp.cpp', 'src/cpp/ptdalgorithmscpp.h']),

        # ('include', ['ptdalgorithms/ptdalgorithms.h']),
        # ('include/ptdalgorithms/api/c', glob('api/c/*.h`') + glob('api/c/*.c`')),
        # ('include/ptdalgorithms/api/cpp', glob('api/cpp/*.h`') + glob('api/cpp/*.cpp`')),
        # ('include/ptdalgorithms/src/c', glob('src/c/*.h`') + glob('src/c/*.c`')),
        # ('include/ptdalgorithms/src/cpp', glob('src/cpp/*.h`') + glob('src/cpp/*.cpp`')),
        # ('include/api', [f for f in glob('src/**', recursive=True) if os.path.isfile(f)]),
        # ('include/src', [f for f in glob('api/**', recursive=True) if os.path.isfile(f)]),
                  ],
     )
#include "../api/c/ptdalgorithms.h"

#include "../src/c/ptdalgorithms.c"
#include "../api/cpp/ptdalgorithmscpp.h"