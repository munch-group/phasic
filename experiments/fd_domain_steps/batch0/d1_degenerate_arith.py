"""Plan A Batch 0, §4.5 degenerate values: probe arithmetic only (no forward).

(a) the zero-probe-difference band of the capped positive rule at large theta
    (plan: "theta above ~2e9, where the capped u falls below ~5.5e-17"), in
    numpy and in jnp under jit, for s = 1e-4;
(b) the same for the full rule (u = s);
(c) the capped positive rule near 0 (probes stay > 0) and the capped unit
    rule near 0 and 1 (probes stay inside (0, 1));
(d) the up-probe overshoot of the capped rules (plan §4.2: never more than
    1e-7 above theta)."""
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

s = 1e-4


def cap_u(t):
    return np.minimum(np.log1p(1e-7 / t), s)


@jax.jit
def cap_probes_j(t):
    u = jnp.minimum(jnp.log1p(1e-7 / t), s)
    return t * jnp.exp(u), t * jnp.exp(-u)


print("(a) capped positive rule, s=1e-4: where do the probes collapse onto theta?")
grid = np.geomspace(1e8, 1e12, 4001)
up_np = grid * np.exp(cap_u(grid)); dn_np = grid * np.exp(-cap_u(grid))
up_j, dn_j = map(np.asarray, cap_probes_j(jnp.asarray(grid)))
for lab, up, dn in (("numpy", up_np, dn_np), ("jnp+jit", up_j, dn_j)):
    upz = grid[up == grid]; dnz = grid[dn == grid]; both = grid[(up == grid) & (dn == grid)]
    diffz = grid[(up - dn) == 0]
    print(f"  {lab:8s} first theta with up==theta: {upz.min() if upz.size else None:.3e}; "
          f"dn==theta: {dnz.min() if dnz.size else None:.3e}; both: "
          f"{both.min() if both.size else None:.3e}; up-dn==0 from {diffz.min() if diffz.size else None:.3e}")
    # fraction of grid points above 2e9 with zero difference
    sel = grid >= 2e9
    print(f"           above 2e9: {np.mean((up - dn)[sel] == 0):.3f} of points have a zero difference; "
          f"between 1e9 and 2e9: {np.mean((up - dn)[(grid >= 1e9) & (grid < 2e9)] == 0):.3f}")
    print(f"           u at 2e9 = {cap_u(2e9):.3e}; ulp(1)/2 = {np.finfo(float).eps/2:.3e}")

print("(b) full positive rule (u = s = 1e-4) never collapses: "
      f"max theta tested 1e300 -> diff {1e300*np.exp(s) - 1e300*np.exp(-s):.3e}")

print("(c) near-zero / near-one probes")
for t in (1e-300, 1e-30, 1e-13, 1e-12, 1e-9, 5e-8, 1e-7):
    u = cap_u(t)
    print(f"  positive capped theta={t:.0e}: u={u:.3e} up={t*np.exp(u):.3e} dn={t*np.exp(-u):.3e} "
          f"(dn>0: {t*np.exp(-u) > 0}; up-theta={t*np.exp(u)-t:.2e})")
sig = lambda x: 1 / (1 + np.exp(-x))
for t in (1e-12, 1e-9, 0.5, 1 - 1e-9, 1 - 1e-12, 1 - 1e-15):
    z = np.log(t / (1 - t))
    h = min(s, 1e-7 / (t * (1 - t))) / (1 + s)
    up, dn = sig(z + h), sig(z - h)
    print(f"  unit capped theta={t!r}: h={h:.3e} up={up!r} dn={dn!r} inside(0,1): "
          f"{0 < dn < 1 and 0 < up < 1}; up-theta={up - t:.2e}; diff==0: {up == dn}")

print("(d) max up-probe overshoot of the capped positive rule over theta in [1e-12, 1e12]:")
g = np.geomspace(1e-12, 1e12, 200001)
ov = g * np.exp(cap_u(g)) - g
print(f"  max(up - theta) = {ov.max():.6e} (bound 1e-7); argmax theta = {g[ov.argmax()]:.3e}")
gu = np.linspace(1e-6, 1 - 1e-6, 200001)
zz = np.log(gu / (1 - gu)); hh = np.minimum(s, 1e-7 / (gu * (1 - gu))) / (1 + s)
ovu = sig(zz + hh) - gu
print(f"  unit rule: max(up - theta) = {ovu.max():.6e} (bound ~1e-7)")

print("(e) where the capped up-probe first exceeds theta + 1e-7 (rounding: ulp(theta) ~ 1e-7)")
g = np.geomspace(1e6, 2e9, 400001)
ov = g * np.exp(cap_u(g)) - g
bad = g[ov > 1e-7]
print(f"  first theta with up - theta > 1e-7: {bad.min():.4e} (ulp there {np.spacing(bad.min()):.3e}); "
      f"fraction of grid points in [that, 2e9] over the bound: {np.mean(ov[g >= bad.min()] > 1e-7):.3f}")
