"""reviewS s4: read-only AST scan of docs notebooks (main tree) and tests/pytest for
NESTED functions that reference themselves or a sibling nested function (would be a
refused cycle under v4.4 if captured by a cached callback), and for callbacks passed
to Graph(...) that are nested functions. Notebooks are read as JSON, never executed."""
import ast, json, glob, os, re
roots_nb = glob.glob("/Users/kmt/phasic/docs/pages/**/*.ipynb", recursive=True)
roots_py = glob.glob("/Users/kmt/phasic-bug5/tests/pytest/*.py")

def sources():
    for p in roots_nb:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        for i, c in enumerate(d.get("cells", [])):
            if c.get("cell_type") != "code":
                continue
            src = "".join(c["source"])
            src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
            yield f"{os.path.relpath(p, '/Users/kmt/phasic/docs/pages')}[{i}]", src
    for p in roots_py:
        yield os.path.basename(p), open(p, encoding="utf-8").read()

hits = []
for where, src in sources():
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for outer in ast.walk(tree):
        if not isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        nested = [n for n in ast.walk(outer) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not outer]
        names = {n.name for n in nested}
        for n in nested:
            used = {x.id for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
            rec = used & names
            if rec:
                hits.append((where, outer.name, n.name, tuple(sorted(rec))))
seen = set()
for h in hits:
    if h in seen: continue
    seen.add(h)
    print(h)
print("total", len(seen))
