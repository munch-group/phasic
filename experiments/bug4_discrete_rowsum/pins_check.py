"""Bug 4 §2.6 pin procedure: diff freshly built discretize pin records
(fix build) against tests/pytest/data/discretize_pins.json, field by field.
Permitted: 'pdf' entries only, |new - old| <= 5e-16, no new value negative.
Prints every differing entry (old/new hex) and PERMITTED or STOP."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests", "pytest"))
import _discretize_pins as pins

old = json.load(open(pins.RECORDS))
new = json.loads(json.dumps(pins.build_records(), sort_keys=True))
diffs = []

def walk(a, b, path):
    if type(a) is not type(b):
        diffs.append((path, a, b)); return
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append((path + [k], a.get(k), b.get(k)))
            else:
                walk(a[k], b[k], path + [k])
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append((path, a, b)); return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + [i])
    elif isinstance(a, float):
        if a.hex() != b.hex():
            diffs.append((path, a, b))
    elif a != b:
        diffs.append((path, a, b))

walk(old, new, [])
ok = True
for path, a, b in diffs:
    is_pdf = len(path) >= 2 and path[-2] == "pdf" and isinstance(path[-1], int)
    good = (is_pdf and isinstance(a, float) and isinstance(b, float)
            and abs(b - a) <= 5e-16 and b >= 0)
    ok &= good
    fa = a.hex() if isinstance(a, float) else repr(a)
    fb = b.hex() if isinstance(b, float) else repr(b)
    print(f"{'ok  ' if good else 'BAD '} {'/'.join(map(str, path))}: {fa} ({a!r}) -> {fb} ({b!r})")
neg_old = [(p, a) for p, a, b in diffs if isinstance(a, float) and a < 0]
print(f"{len(diffs)} differing entries; {len(neg_old)} of them negative before")
print("PERMITTED" if ok else "STOP")
if ok and "--write" in sys.argv:
    with open(pins.RECORDS, "w") as f:
        json.dump(pins.build_records(), f, indent=1, sort_keys=True)
    print("regenerated", pins.RECORDS)
