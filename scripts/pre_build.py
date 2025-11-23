import os
import sys

def setup_cmake_prefix_path():
    """Set CMAKE_PREFIX_PATH to help CMake find conda packages."""
    prefixes = []

    # Conda/pixi build environment
    if "BUILD_PREFIX" in os.environ:
        prefixes.append(os.environ["BUILD_PREFIX"])

    # Conda/pixi target environment
    if "PREFIX" in os.environ:
        prefixes.append(os.environ["PREFIX"])

    # Python's prefix (for pip install)
    prefixes.append(sys.prefix)
    prefixes.append(sys.exec_prefix)

    # Join with system path separator and set
    if prefixes:
        existing = os.environ.get("CMAKE_PREFIX_PATH", "")
        new_paths = os.pathsep.join(prefixes)
        if existing:
            os.environ["CMAKE_PREFIX_PATH"] = f"{new_paths}{os.pathsep}{existing}"
        else:
            os.environ["CMAKE_PREFIX_PATH"] = new_paths
        print(f"CMAKE_PREFIX_PATH={os.environ['CMAKE_PREFIX_PATH']}")

def fix_eigen():
    """Create Eigen symlink for compatibility."""
    def symlink_eigen(prefix):
        target_path = f"{prefix}/include/eigen3/Eigen"
        link_path = f"{prefix}/include/Eigen"
        try:
            os.symlink(target_path, link_path)
        except FileExistsError:
            pass
        except Exception:
            pass  # Silently ignore errors

    if "BUILD_PREFIX" in os.environ and os.environ["BUILD_PREFIX"]:
        symlink_eigen(os.environ["BUILD_PREFIX"])

    if "PREFIX" in os.environ and os.environ["PREFIX"]:
        symlink_eigen(os.environ["PREFIX"])

    symlink_eigen(sys.exec_prefix)

if __name__ == "__main__":
    setup_cmake_prefix_path()
    fix_eigen()


