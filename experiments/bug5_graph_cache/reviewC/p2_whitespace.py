"""Review C p2: which whitespace-only edits change co_code on this Python
(plan 2.2.1 says comment and whitespace edits hit)."""
import sys, dis
print(sys.version.split()[0])
base = "def g(s, k=2):\n    r = s + k * 3\n    return [r, (s, k)]\n"
variants = {
    "blank lines + comment": "def g(s, k=2):\n\n    # c\n    r = s + k * 3\n\n    return [r, (s, k)]\n",
    "indent 2 spaces": "def g(s, k=2):\n  r = s + k * 3\n  return [r, (s, k)]\n",
    "spaces around ops": "def g(s,k=2):\n    r=s+k*3\n    return [r,(s,k)]\n",
    "split binop across lines": "def g(s, k=2):\n    r = (s +\n         k * 3)\n    return [r, (s, k)]\n",
    "split tuple across lines": "def g(s, k=2):\n    r = s + k * 3\n    return [r, (s,\n  k)]\n",
    "split call args": "def g(s, k=2):\n    r = s + k * 3\n    return list(\n        [r, (s, k)])\n",
}
def code(src):
    ns = {}
    exec(compile(src, "<x>", "exec"), ns)
    return ns["g"].__code__
cb = code(base)
for name, src in variants.items():
    c = code(src)
    print(f"{name:28s} co_code equal: {c.co_code == cb.co_code}")
c = code(variants["split binop across lines"])
print("--- base"); dis.dis(cb)
print("--- split binop"); dis.dis(c)
