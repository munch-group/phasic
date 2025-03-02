
c++ -O3 -Wall -undefined dynamic_lookup -shared -std=c++11 -fPIC $(python -m pybind11 --includes) api/cpp/ptdalgorithmscpp_pybind.cpp -o ptdalgorithmscpp$(python3-config --extension-suffix)

conda env config vars set CPATH=${CONDA_PREFIX}/include:${CPATH}

conda env config vars set CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/include:${CPLUS_INCLUDE_PATH}

