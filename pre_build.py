import os
import sys

def fix_eigen():

    def symlink_eigen(prefix):
        target_path = f"{prefix}/include/eigen3/Eigen"
        link_path = f"{prefix}/include/Eigen" 
        try:
            os.symlink(target_path, link_path)
        except FileExistsError:
            pass

        if "BUILD_PREFIX" in os.environ and os.environ["BUILD_PREFIX"]:
            symlink_eigen(os.environ["BUILD_PREFIX"])

        if "PREFIX" in os.environ and os.environ["PREFIX"]:
            symlink_eigen(os.environ["PREFIX"])

        symlink_eigen(sys.exec_prefix)

if __name__ == "__main__":
    fix_eigen()


