"""The discretized (was_dph) coalescent at small rates: s1 found a Richardson
spread of 3.8 at theta = [1e-5, 3e-5]. A was_dph pmf depends on rate RATIOS
only, so (1) F(c * theta) must not depend on c; (2) F must be smooth in a slot.
Same model as s1 (coalescent_graph(5).discretize(1.0), jumps 0..8)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from s1_step_sizes import _nmr

g = _nmr().coalescent_graph(5).discretize(1.0)
m = Graph.pmf_from_graph(g, discrete=True, theta_dim=2)
jumps = jnp.asarray([0.0, 1.0, 2.0, 3.0, 5.0, 8.0])
F = lambda th: np.asarray(m(jnp.asarray(th), jumps))
np.set_printoptions(precision=6, linewidth=160)
print("(1) scale invariance, ratio t1/t0 = 3")
for c in (1.0, 1e-2, 1e-3, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-7):
    print(f"   theta = [{c:g}, {3*c:g}]: pmf = {F([c, 3 * c])}  sum={F([c, 3*c]).sum():.6f}")
print("(2) smoothness in slot 0 around [1e-5, 3e-5], relative offsets x")
base = F([1e-5, 3e-5])
for x in (-1e-2, -1e-3, -1e-4, -1e-5, 0.0, 1e-5, 1e-4, 1e-3, 1e-2):
    print(f"   x={x:+.0e}: pmf/pmf0 - 1 = {F([1e-5 * (1 + x), 3e-5]) / base - 1}")
