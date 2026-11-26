set -e
cd /Users/kmt/phasic/experiments/core_ssot/review20
PY=/Users/kmt/phasic/.pixi/envs/default/bin/python
PD=$($PY -c "import phasic,os;print(os.path.dirname(phasic.__file__))")
INC="$($PY -c "import pybind11,sysconfig;print('-I'+pybind11.get_include()+' -I'+sysconfig.get_paths()['include'])") -I$PD/include/c -I$PD/include/cpp"
SUF=$($PY -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
CXX=/Users/kmt/phasic/.pixi/envs/default/bin/clang++
[ -x $CXX ] || CXX=c++
$CXX -std=c++17 -O1 -shared -fPIC $INC -DMODNAME=mod_ref mod_ref.cpp -undefined dynamic_lookup -o mod_ref$SUF
$CXX -std=c++17 -O1 -shared -fPIC $INC -DMODNAME=mod_anchor -DANCHOR mod_ref.cpp -undefined dynamic_lookup -o mod_anchor$SUF
ls *.so
