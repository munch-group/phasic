# Literal 2*eps divisor vs actual-difference divisor for a plain +-1e-7 probe,
# f(x) = log(x) style smooth function; exact derivative known.
import math
eps = 1e-7
print("theta   rel_err_literal   rel_err_actual")
for th in [1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 3e9]:
    f = lambda x: x * x * 0.5          # f' = x
    xp, xm = th + eps, th - eps
    exact = th
    num = f(xp) - f(xm)
    lit = num / (2 * eps)
    act = num / (xp - xm) if xp != xm else 0.0
    print(f"{th:8.0e}  {abs(lit-exact)/exact:.3e}  {abs(act-exact)/exact:.3e}  actual_diff={xp-xm:.4e}")
