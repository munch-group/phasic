"""Review F probes (run with HOME=scratch, OMP_NUM_THREADS unset).

a) who imports phasic.phasic_pybind first, and is OMP_NUM_THREADS set then
b) PropertySet vars() fresh vs after index_to_props (allow-list pin)
c) StateIndexer keyed by to_dict(): aliasing of the caller's property list
d) numpy StringDType: hasobject / fields / tobytes collision
"""
import os, sys, traceback

first = {}


class Hook:
    def find_spec(self, name, path=None, target=None):
        if name == "phasic.phasic_pybind" and not first:
            first["omp_set"] = "OMP_NUM_THREADS" in os.environ
            st = traceback.extract_stack()
            first["stack"] = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno}"
                              for f in st if "phasic" in f.filename][:12]
        return None


sys.meta_path.insert(0, Hook())
print("OMP_NUM_THREADS in env before import:", "OMP_NUM_THREADS" in os.environ)
import phasic  # noqa: E402
print("(a) first phasic_pybind import: OMP set =", first.get("omp_set"))
for s in first.get("stack", []):
    print("    ", s)

from phasic.state_indexing import StateIndexer, PropertySet, Property  # noqa: E402

# b) PropertySet attribute set
ix = StateIndexer(lineage=[Property("a", max_value=3)])
ps = ix.lineage
print("(b) fresh PropertySet vars:", sorted(vars(ps)))
ix.lineage.index_to_props(1)
print("    after index_to_props:  ", sorted(vars(ps)))
print("    fresh StateIndexer vars:", sorted(vars(ix)))

# c) aliasing: caller's list mutated after the indexer was built
props = [Property("a", max_value=3)]
ixA = StateIndexer(lineage=props)
props[0] = Property("a", max_value=5)          # caller reuses its list
ixB = StateIndexer(lineage=[Property("a", max_value=5)])
print("(c) to_dict equal:", ixA.to_dict() == ixB.to_dict(),
      "| vars sets equal:", set(vars(ixA)) == set(vars(ixB)))
print("    state_length A, B:", ixA.state_length, ixB.state_length)
for _n, _ix in (("A", ixA), ("B", ixB)):
    try:
        print("    %s.lineage.index_to_props(4):" % _n, _ix.lineage.index_to_props(4, as_dict=True))
    except Exception as e:
        print("    %s.lineage.index_to_props(4): %s %s" % (_n, type(e).__name__, e))
props2 = [Property("a", max_value=3)]
ixC = StateIndexer(lineage=props2)
props2.append(Property("b", max_value=1))      # append to the same list
ixD = StateIndexer(lineage=[Property("a", max_value=3), Property("b", max_value=1)])
print("    append case: to_dict equal:", ixC.to_dict() == ixD.to_dict(),
      "state_length C, D:", ixC.state_length, ixD.state_length)

# d) StringDType
import numpy as np  # noqa: E402
print("(d) numpy", np.__version__)
try:
    from numpy.dtypes import StringDType
    a = np.array(["x" * 40], dtype=StringDType())
    b = np.array(["y" * 40], dtype=StringDType())
    print("    dtype.str", a.dtype.str, "hasobject", a.dtype.hasobject,
          "fields", a.dtype.fields, "type is ndarray", type(a) is np.ndarray)
    print("    tobytes equal for different 40-char strings:", a.tobytes() == b.tobytes())
    s1 = np.array(["short1"], dtype=StringDType()); s2 = np.array(["short2"], dtype=StringDType())
    print("    tobytes equal for different short strings:", s1.tobytes() == s2.tobytes())
except Exception as e:
    print("    StringDType probe error:", type(e).__name__, e)
