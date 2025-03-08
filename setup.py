from glob import glob
import os
import sys
import platform
import subprocess
from setuptools import setup, find_packages

version = "0.1.24"

# if 'PREFIX' in os.environ and os.environ["PREFIX"]:
#     prefix = os.environ["PREFIX"]
# else:
#     prefix = sys.exec_prefix
prefix = sys.exec_prefix

from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, build_ext, naive_recompile
# from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, naive_recompile
# from setuptools.command.build_ext import build_ext as _build_ext
# class build_ext(_build_ext):
#     def run(self):
#         # # Run the pre-build command
#         pre_build_command = "python pre_build.py"
#         subprocess.check_call(pre_build_command, shell=True)
#         assert 0

#         # Continue with the normal build process
#         _build_ext.run(self)

# Optional multithreaded build
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

ParallelCompile("NPY_NUM_BUILD_JOBS", needs_recompile=naive_recompile).install()

# extra_compile_args=["-g", f"-I{prefix}/include/eigen3/"]
extra_compile_args=["-g", "-I${PREFIX}/include/eigen3/"]

# extra_link_args = ["-g", f"-I{prefix}/include/eigen3/"]
extra_link_args = ["-g", "-I${PREFIX}/include/eigen3/"]
if platform.system() == "Darwin":
  # Compiling on macOS requires an installation of the Xcode Command Line Tools
  # os.environ["CC"] = "g++"
  # os.environ["CXX"] = "g++ -std=c++11"
  os.environ["CC"] = "g++ -g -I${PREFIX}/include/eigen3/"
  os.environ["CXX"] = "g++ -g -I${PREFIX}/include/eigen3/ -std=c++11"


  # extra_link_args.extend(["-undefined", "dynamic_lookup"])

#include/eigen3

ext_modules = [
    Pybind11Extension(
        "ptdalgorithms.ptdalgorithmscpp_pybind",
        sorted(glob("src/*/*.cpp") + glob("src/*/*.c")),
#        cxx_std=11,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[f"{prefix}/include/", f"{prefix}/include/eigen3/"],
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
      extras_require={"test": "pytest"},
      cmdclass={"build_ext": build_ext}, 
      ext_modules=ext_modules,
      data_files=[
        ('include', ['ptdalgorithms/ptdalgorithms.h']),
        ('include/ptdalgorithms/api/c', ['api/c/ptdalgorithms.h']),
        ('include/ptdalgorithms/api/cpp', ['api/cpp/ptdalgorithmscpp.h']),
        ('include/ptdalgorithms/src/c', ['src/c/ptdalgorithms.c', 'src/c/ptdalgorithms.h']),
        ('include/ptdalgorithms/src/cpp', ['src/cpp/ptdalgorithmscpp.cpp', 'src/cpp/ptdalgorithmscpp.h', 'src/cpp/ptdalgorithmscpp_pybind.cpp']),
      ])