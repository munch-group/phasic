"""reviewO p4: v4.0 §2.2.2 re-entry budget (10^6) read literally:
'walking a container whose id() was already walked earlier in the same key
computation counts against a budget of 10^6'. Count re-entries for ordinary
lookup tables built from canonical state tuples (shared objects).
No phasic import; pure Python walk mirroring the rule."""
import itertools, sys

def count_reentries(root):
    walked, reentries, stack = set(), 0, [root]
    while stack:
        x = stack.pop()
        if type(x) in (list, tuple, dict):
            if id(x) in walked:
                reentries += 1
            walked.add(id(x))
            if type(x) is dict:
                for k, v in x.items():
                    stack.append(k); stack.append(v)
            else:
                stack.extend(x)
    return reentries

print("empty tuple is a singleton:", tuple([]) is tuple())

# 1. adjacency table over canonical state tuples (n states, out-degree d)
for n, d in ((20_000, 10), (60_000, 20), (200_000, 6)):
    states = [(i % 97, i // 97, i % 7) for i in range(n)]   # canonical tuples
    adj = {s: [states[(i * 31 + j) % n] for j in range(d)]
           for i, s in enumerate(states)}
    r = count_reentries(adj)
    print(f"adjacency n={n:>7} d={d:>2}: re-entries={r:>9}  refused={r > 10**6}")

# 2. table with many empty-tuple values (absorbing states -> ())
n = 600_000
tab = {i: () for i in range(n)}
r = count_reentries(tab)
print(f"{n} int keys -> (): re-entries={r}  refused={r > 10**6}")
tab2 = {(i,): ((), ()) for i in range(n)}
r = count_reentries(tab2)
print(f"{n} keys -> ((), ()): re-entries={r}  refused={r > 10**6}")

# 3. literal tuple constants are one shared object per code location
def build(n):
    return {i: {"init": (0, 1), "absorb": (0, 0)} for i in range(n)}
t = build(3)
print("literal (0, 1) shared across entries:", t[0]["init"] is t[2]["init"])
r = count_reentries(build(600_000))
print(f"600000 entries with two literal tuple fields: re-entries={r}  refused={r > 10**6}")
