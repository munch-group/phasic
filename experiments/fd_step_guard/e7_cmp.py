import numpy as np
D='/private/tmp/claude-501/-Users-kmt-phasic/3666ba56-7958-4c29-a8b2-ec58b11551e2/scratchpad/fd/'
a, b = np.load(D+'e7_old.npz'), np.load(D+'e7_new.npz')
for k in a.files:
    if k.endswith('/time'): continue
    x, y = a[k], b[k]
    same = np.array_equal(x, y, equal_nan=True)
    fin_o, fin_n = np.isfinite(x).mean(), np.isfinite(y).mean()
    with np.errstate(all='ignore'):
        rel = np.nanmax(np.abs(x-y)/np.maximum(np.abs(y), 1e-300))
    print(f'{k:28s} bit-identical={same}  finite old={fin_o:.2f} new={fin_n:.2f}  max rel diff={rel:.2e}')
