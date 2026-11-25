"""reviewAF p5: are the per-entry comparisons suggested for finding 1 satisfiable on an ordinary indexer?"""
import os, dataclasses
assert os.environ["HOME"].startswith("/private/tmp/")
from phasic.state_indexing import StateIndexer, Property

ix = StateIndexer(lineages=[Property("ton", min_value=1, max_value=4)], slots=["epoch"])
ix.lineages.index_to_props(2)
c = StateIndexer.from_dict(ix.to_dict())
for label, ref, cand in (("_result_class", c._result_class, ix._result_class),
                         ("_props_dataclass", c.lineages._props_class(), ix.lineages._props_dataclass)):
    ra, ca = vars(ref)["__annotations__"], vars(cand)["__annotations__"]
    print(label, "annotations values identical:", all(ra[k] is ca[k] for k in ra), list(ra) == list(ca))
    rf, cf = ref.__dataclass_fields__, cand.__dataclass_fields__
    attrs = ("name", "type", "default", "default_factory", "init", "repr", "hash", "compare", "kw_only", "_field_type")
    print(label, "fields equal by (identity of) attrs:", list(rf) == list(cf) and all(getattr(rf[k], a) is getattr(cf[k], a) or getattr(rf[k], a) == getattr(cf[k], a) for k in rf for a in attrs))
    rp, cp = ref.__dataclass_params__, cand.__dataclass_params__
    print(label, "params repr equal:", repr(rp) == repr(cp))
    for k in ("__dict__", "__weakref__"):
        print(label, k, "objclass is own class:", vars(ref)[k].__objclass__ is ref, vars(cand)[k].__objclass__ is cand)
    for fn in ("__setattr__", "__delattr__"):
        cells = dict(zip(vars(cand)[fn].__code__.co_freevars, [x.cell_contents for x in vars(cand)[fn].__closure__]))
        print(label, fn, "FrozenInstanceError cell is dataclasses.FrozenInstanceError:", cells["FrozenInstanceError"] is dataclasses.FrozenInstanceError)
    cells = dict(zip(vars(cand)["__repr__"].__code__.co_freevars, [x.cell_contents for x in vars(cand)["__repr__"].__closure__]))
    print(label, "__repr__ repr_running:", type(cells["repr_running"]).__name__, len(cells["repr_running"]))
