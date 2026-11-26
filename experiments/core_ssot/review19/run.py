import sys, os, ctypes, glob, subprocess
case = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if case.startswith("noreopen_"):
    import ext_a
    name = case[len("noreopen_"):]
elif case.startswith("nopre_"):
    name = case[len("nopre_"):]
else:
    import ext_a
    ctypes.CDLL(ext_a.__file__, mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    name = case
try:
    m = __import__(name); print("IMPORTED", name, m.hello())
except BaseException as e:
    print("EXC", type(e).__name__, e, "| cause:", repr(e.__cause__))
