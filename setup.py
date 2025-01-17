from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

from pybind11.setup_helpers import ParallelCompile

# Optional multithreaded build
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

ext_modules = [
    Pybind11Extension(
        "python_example",
        sorted(glob("src/*.cpp")),
    ),
]

setup(..., cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
