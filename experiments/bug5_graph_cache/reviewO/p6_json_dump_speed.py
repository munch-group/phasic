"""reviewO p6: v4.0 §2.6 saves with json.dump(entry, f, separators=(",",":"))
(streamed). CPython's json.dump never uses the C encoder (only dumps does:
JSONEncoder.iterencode uses c_make_encoder only when _one_shot is True).
Measure dump-to-file vs dumps+write on a graph_data-like payload
(lists of float rows, like serialize() arrays after tolist())."""
import json, os, sys, time, tempfile, random
import json.encoder as je
print("python", sys.version.split()[0], " c_make_encoder available:", je.c_make_encoder is not None)
random.seed(0)
rows = 1_000_000                       # ~ edges rows of [from, to, w] + coeffs
payload = {"graph_data": {"edges": [[float(i), float(i + 1), random.random(), random.random()]
                                    for i in range(rows)]}}
d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
p = os.path.join(d, "x.json")
t = time.perf_counter()
with open(p, "w", encoding="utf-8") as f:
    json.dump(payload, f, separators=(",", ":"))
t_dump = time.perf_counter() - t
size = os.path.getsize(p)
t = time.perf_counter()
s = json.dumps(payload, separators=(",", ":"))
with open(p, "w", encoding="utf-8") as f:
    f.write(s)
t_dumps = time.perf_counter() - t
os.remove(p); os.rmdir(d)
print(f"{rows} rows x 4 numbers, file {size/1e6:.1f} MB")
print(f"json.dump  (streamed, pure-Python encoder): {t_dump:.2f} s")
print(f"json.dumps + write (C encoder):             {t_dumps:.2f} s   ratio {t_dump/t_dumps:.1f}x")
