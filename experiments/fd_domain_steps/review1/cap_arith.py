# Arithmetic checks of the plan's transitional cap and log/logit probes (no phasic import).
import numpy as np
s = 1e-4
print("cap u=min(1e-7/theta, s): up-displacement / 1e-7")
for th in [1e-12, 1e-9, 1e-4, 9.99e-4, 1e-3, 1.0001e-3, 1e-2, 1.0, 1e3, 1e8, 1e9, 1e10]:
    u = min(1e-7/th, s)
    tp, tm = th*np.exp(u), th*np.exp(-u)
    print(f"theta={th:9.3g} up={(tp-th)/1e-7:.8f} down={(th-tm)/1e-7:.8f} actual_diff={tp-tm:.3e} zero={tp==tm}")
print("today theta +/- 1e-7 at large theta: probes equal?")
for th in [1e8, 1e9, 1e10]:
    print(th, (th+1e-7)==(th-1e-7), (th+1e-7)-(th-1e-7))
print("bit identity: (t+e)-(t-e) vs 2e")
for th in [0.3, 1.0, 1.7, 5.0]:
    print(th, (th+1e-7)-(th-1e-7), 2*1e-7, ((th+1e-7)-(th-1e-7))==2e-7)
print("theta==0 under positive log rule:", 0.0*np.exp(s), 0.0*np.exp(-s))
def sig(x): return 1/(1+np.exp(-x))
for p in [1e-9, 1-1e-9, 1-1e-15, 1.0]:
    with np.errstate(all='ignore'):
        lg = np.log(p) - np.log1p(-p)
        print("unit p=",p, sig(lg+s), sig(lg-s))
