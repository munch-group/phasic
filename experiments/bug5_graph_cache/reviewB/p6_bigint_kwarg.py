"""reviewB p6: Graph.__init__ converts every int keyword argument to float
before calling the callback (src/phasic/__init__.py:1743-1745). An int above
4300 digits as a TOP-LEVEL kwarg therefore cannot build (test 12 wording)."""
v = 10 ** 5000
try:
    float(v)
    print("float() ok")
except OverflowError as e:
    print("OverflowError:", e)
