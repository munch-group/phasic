"""Review 7 probes (master code; HOME redirected)."""
import os, sys, json, pathlib, stat, tempfile, io, contextlib
assert "scratchpad" in os.environ["HOME"]
assert not os.environ.get("PHASIC_CACHE_DIR")
HOME = pathlib.Path(os.environ["HOME"])


def sec(t):
    print("\n=== " + t, flush=True)


sec("A. _deserialize_value on graph_data imports an arbitrary module named in the entry")
from phasic.graph_cache import _deserialize_value
planted = {"__type__": "this.NoSuchClass", "__data__": {}}
buf = io.StringIO()
was_loaded = "this" in sys.modules
with contextlib.redirect_stdout(buf):
    try:
        _deserialize_value(planted)
        res = "returned"
    except Exception as e:
        res = f"{type(e).__name__}: {str(e)[:80]}"
print("  'this' imported before:", was_loaded, "| after:", "this" in sys.modules,
      "| module import side effect printed", len(buf.getvalue()), "chars | result:", res)
planted2 = {"__type__": "collections.OrderedDict", "__data__": {"a": 1}}
print("  arbitrary installed class .from_dict called? ->", type(_deserialize_value(planted2)).__name__
      if hasattr(__import__("collections").OrderedDict, "from_dict") else "OrderedDict has no from_dict (TypeError path)")
import pandas as pd
r = _deserialize_value({"__type__": "pandas.DataFrame", "__data__": {"x": [1, 2]}})
print("  pandas.DataFrame.from_dict invoked from an entry:", type(r).__name__)

sec("B. owner lookup when the directory is not searchable")
d = pathlib.Path(tempfile.mkdtemp(dir=HOME)); f = d / "e.json"; f.write_text("{}")
os.chmod(d, 0o600)  # no x bit: open() and stat() both fail
for op, fn in (("open", lambda: open(f).close()), ("os.stat", lambda: os.stat(f))):
    try:
        fn(); print(f"  {op}: ok")
    except Exception as e:
        print(f"  {op}: {type(e).__name__}")
os.chmod(d, 0o700); f.unlink(); d.rmdir()
import pwd
try:
    pwd.getpwuid(987654)
except KeyError as e:
    print("  pwd.getpwuid(unknown uid) -> KeyError (plan: falls back to uid)")

sec("C. leftover .partial files are not removed by clear_graph_cache (glob *.json)")
from phasic.graph_cache import GraphCache
gd = HOME / "gc_probe"; gc = GraphCache(gd)
(gd / ".tmp-deadbeefdeadbeef.partial").write_text("x")
n = gc.clear_graph_cache()
print("  removed:", n, "| .partial left:", [p.name for p in gd.iterdir()])
for p in gd.iterdir(): p.unlink()
gd.rmdir()
