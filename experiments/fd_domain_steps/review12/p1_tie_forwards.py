# Which forwards does the v2.7 §4.5 tie fallback need? Offset slot t beside an O(1) rate,
# f(t) = (1+t) exp(-(1+t) x), x = 1.3 (exponential pdf). Capped band rule (v2.7 §4.2):
# u = min(log1p(1e-7/t), s), s = 1e-4.
import numpy as np
s = 1e-4; x = 1.3
f = lambda t: (1+t)*np.exp(-(1+t)*x)
for t in [5e-8, 1e-9, 1e-11, 1e-13]:
    u = min(np.log1p(1e-7/t), s)
    up, dn = t*np.exp(u), t*np.exp(-u)
    tie = f(up) == f(dn)
    print(f"t={t:.0e} u={u:.1e} up={up:.6e} (t+1e-7={t+1e-7:.6e}, equal={up==t+1e-7}) "
          f"tie={tie} f(t)==f(up)={f(t)==f(up)} onesided[t,t+1e-7]={(f(t+1e-7)-f(t))/1e-7:.6e} "
          f"exact={(1-(1+t)*x)*np.exp(-(1+t)*x):.6e}")
