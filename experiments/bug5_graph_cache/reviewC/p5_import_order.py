"""Review C p5: the import-order check of plan test 14, on master.
Child: OMP_NUM_THREADS removed; a meta_path hook reports whether it is set
when phasic.phasic_pybind is first imported, and which OpenMP images are
already loaded at that moment (macOS dyld). Also a negative control: a stub
package that imports phasic.cache first."""
import os, subprocess, sys, textwrap, tempfile, pathlib

CHILD = textwrap.dedent(r'''
import os, sys, ctypes
def omp_images():
    try:
        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        n = dyld._dyld_image_count()
        return sorted({os.path.basename(dyld._dyld_get_image_name(i).decode())
                       for i in range(n) if b"omp" in dyld._dyld_get_image_name(i).lower()})
    except Exception as e:
        return ["?" + repr(e)]
class Hook:
    seen = False
    def find_spec(self, name, path, target=None):
        if name == "phasic.phasic_pybind" and not Hook.seen:
            Hook.seen = True
            print("AT_PYBIND OMP_NUM_THREADS=", os.environ.get("OMP_NUM_THREADS"),
                  "omp images already loaded:", omp_images(), flush=True)
        return None
sys.meta_path.insert(0, Hook())
MODE = sys.argv[1]
if MODE == "real":
    import phasic
elif MODE == "stub":
    import phasic   # resolves to the stub on sys.path[0]
print("END omp images:", omp_images(), "OMP=", os.environ.get("OMP_NUM_THREADS"))
''')

def run(mode, extra_path=None):
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    if extra_path:
        env["PYTHONPATH"] = extra_path
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(CHILD)
        p = f.name
    r = subprocess.run([sys.executable, p, mode], env=env, capture_output=True, text=True)
    os.unlink(p)
    print("==", mode, "rc", r.returncode)
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith(("AT_", "END"))))
    if r.returncode:
        print(r.stderr[-1500:])

run("real")

# negative control: stub package named phasic whose __path__ is the real
# package dir, importing .cache before anything sets OMP_NUM_THREADS
import importlib.util
real_dir = os.path.dirname(importlib.util.find_spec("phasic").origin)
stub_root = tempfile.mkdtemp(dir=os.environ.get("REVIEWC_SCRATCH"))
pkg = pathlib.Path(stub_root, "phasic")
pkg.mkdir()
(pkg / "__init__.py").write_text(f"__path__ = [{real_dir!r}]\nfrom . import cache\n")
run("stub", stub_root)
import shutil; shutil.rmtree(stub_root)
