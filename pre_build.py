import os

PREFIX = os.getenv('PREFIX')
target_path = f"{PREFIX}/include/eigen3/Eigen"
link_path = f"{PREFIX}/include/Eigen" 
try:
    os.symlink(target_path, link_path)
except FileExistsError:
    pass

BUILD_PREFIX = os.getenv('BUILD_PREFIX')
target_path = f"{BUILD_PREFIX}/include/eigen3/Eigen"
link_path = f"{BUILD_PREFIX}/include/Eigen" 
try:
    os.symlink(target_path, link_path)
except FileExistsError:
    pass
