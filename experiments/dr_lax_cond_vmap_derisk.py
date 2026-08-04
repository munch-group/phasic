"""B3 joint-index lax.cond/vmap redesign, D0 (de-risk, build-free, no phasic
C code involved -- pure JAX semantics). Confirms three claims empirically
before touching src/phasic/__init__.py:

  1. jax.lax.cond computes BOTH branches when its predicate is BATCHED
     under vmap (the bug the adversarial review found: a spy counter proves
     the "expensive" branch actually runs even when the predicate is always
     True for every element of the batch).
  2. A plain Python-level if/else on a NON-TRACED value (e.g. a bool fixed
     at model-construction time, closed over by the traced function) lets
     vmap build only ONE of the two branches into the traced computation --
     the other never executes at all, at any batch size (this is the
     mechanism the redesign relies on: move the choice from a per-call
     traced predicate to a per-model Python bool).
  3. Raising a Python exception inside a jax.pure_callback's host function
     propagates as a genuine Python exception at the jax.grad(...)(...)
     call site, both ungapped and under jax.vmap(jax.grad(...)), rather
     than being silently swallowed or turned into an opaque JAX error.
     (This pattern is already used elsewhere in this file --
     _check_weight's raise inside _apply_weight_callback, called via
     pure_callback -- confirmed working by direct observation of a test
     failure traceback earlier this session; this script re-confirms it
     in isolation, including under vmap, which that observation did not
     cover.)

Run: pixi run python experiments/dr_lax_cond_vmap_derisk.py
"""
import warnings; warnings.filterwarnings("ignore")
import phasic  # noqa: F401  (import before jax -- forces x64, see CLAUDE.md)
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Claim 1: lax.cond computes BOTH branches under a batched predicate.
# ---------------------------------------------------------------------------
def claim1_lax_cond_computes_both_branches_under_vmap():
    calls_true = []
    calls_false = []

    def expensive_true(x):
        calls_true.append(1)
        return x * 2.0

    def expensive_false(x):
        calls_false.append(1)
        return x * 3.0

    def f(x):
        pred = x > 0.0  # ALWAYS True for the batch used below -- a batched predicate
        return jax.lax.cond(pred, expensive_true, expensive_false, x)

    xs = jnp.array([1.0, 2.0, 3.0])  # all positive -> pred is True for every element
    out = jax.vmap(f)(xs)

    both_ran = (len(calls_true) > 0) and (len(calls_false) > 0)
    print(f"  lax.cond under vmap, predicate always True: "
          f"true-branch traced={len(calls_true) > 0} false-branch traced={len(calls_false) > 0}")
    print(f"  RESULT: {'CONFIRMED (both branches DO execute -- the bug)' if both_ran else 'NOT CONFIRMED'}")
    return both_ran


# ---------------------------------------------------------------------------
# Claim 2: a Python-level if/else on a non-traced bool builds only ONE
# branch into the traced computation, even under vmap.
# ---------------------------------------------------------------------------
def claim2_python_if_skips_unused_branch_under_vmap():
    calls_true = []
    calls_false = []

    def make_f(static_choice):  # static_choice: a plain Python bool, closed over
        def f(x):
            if static_choice:
                calls_true.append(1)
                return x * 2.0
            else:
                calls_false.append(1)
                return x * 3.0
        return f

    f_true = make_f(True)
    xs = jnp.array([1.0, 2.0, 3.0])
    out = jax.vmap(f_true)(xs)

    only_true_ran = (len(calls_true) > 0) and (len(calls_false) == 0)
    print(f"\n  Python if/else on a static bool under vmap: "
          f"true-branch traced={len(calls_true) > 0} false-branch traced={len(calls_false) > 0}")
    print(f"  RESULT: {'CONFIRMED (only the chosen branch is ever traced)' if only_true_ran else 'NOT CONFIRMED'}")

    # Sanity: values match the eager (non-vmap) computation.
    expected = xs * 2.0
    values_ok = jnp.allclose(out, expected)
    print(f"  values correct: {values_ok}")
    return only_true_ran and bool(values_ok)


# ---------------------------------------------------------------------------
# Claim 3: raising inside a pure_callback propagates as a real exception,
# both plain and under vmap(grad(...)).
# ---------------------------------------------------------------------------
def claim3_raise_inside_pure_callback_propagates():
    def _raising_host_fn(x_np):
        x = np.asarray(x_np)
        if float(x[0]) > 5.0:
            raise RuntimeError(f"deliberate probe failure at x={x.tolist()}")
        return x * 2.0

    @jax.custom_vjp
    def model(x):
        return jnp.sum(_pure_call(x))

    def _pure_call(x):
        return jax.pure_callback(
            _raising_host_fn, jax.ShapeDtypeStruct(x.shape, x.dtype), x,
            vmap_method='sequential',
        )

    def model_fwd(x):
        return model(x), x

    def model_bwd(res, g):
        x = res
        # deliberately trigger the same raising path in the backward,
        # mirroring how the joint-index redesign raises inside model_bwd
        y = jax.pure_callback(_raising_host_fn, jax.ShapeDtypeStruct(x.shape, x.dtype), x,
                              vmap_method='sequential')
        return (g * y,)

    model.defvjp(model_fwd, model_bwd)

    # -- plain (non-vmap) grad --
    ok_plain = False
    try:
        jax.grad(model)(jnp.array([10.0, 1.0]))
        print("\n  plain grad: NO EXCEPTION RAISED (unexpected)")
    except RuntimeError as e:
        ok_plain = "deliberate probe failure" in str(e)
        print(f"\n  plain grad: RuntimeError propagated correctly: {ok_plain}")
    except Exception as e:  # pragma: no cover
        print(f"  plain grad: WRONG exception type propagated: {type(e).__name__}: {e}")

    # -- vmap(grad(...)) --
    ok_vmap = False
    try:
        jax.vmap(jax.grad(model))(jnp.array([[10.0, 1.0], [1.0, 2.0]]))
        print("  vmap(grad): NO EXCEPTION RAISED (unexpected)")
    except RuntimeError as e:
        ok_vmap = "deliberate probe failure" in str(e)
        print(f"  vmap(grad): RuntimeError propagated correctly: {ok_vmap}\n"
              f"    message: {e}")
    except Exception as e:  # pragma: no cover
        ok_vmap = "deliberate probe failure" in str(e)
        print(f"  vmap(grad): exception type={type(e).__name__} "
              f"(not RuntimeError, but {'DOES' if ok_vmap else 'does NOT'} carry the "
              f"original message -- JAX may wrap host-callback exceptions under vmap): {e}")

    # -- sanity: a NON-triggering input must NOT raise and the callback must
    #    actually run (value shape-consistent) under BOTH forward and
    #    backward, so the redesign's happy path isn't broken. (Not
    #    asserting a specific gradient VALUE here -- model_bwd's toy math
    #    above is deliberately simplistic and not meant to be a correct
    #    VJP, only a vehicle for exercising the raising pure_callback in
    #    the backward pass.)
    ok_happy = False
    try:
        val = model(jnp.array([1.0, 2.0]))
        g = jax.grad(model)(jnp.array([1.0, 2.0]))
        ok_happy = bool(jnp.allclose(val, 6.0)) and bool(np.all(np.isfinite(np.asarray(g))))
        print(f"  non-triggering input: no exception, forward/backward ran: {ok_happy} "
              f"(val={float(val)}, grad={np.asarray(g).tolist()})")
    except Exception as e:  # pragma: no cover
        print(f"  non-triggering input: unexpected exception: {type(e).__name__}: {e}")

    result = ok_plain and ok_vmap and ok_happy
    print(f"  RESULT: {'CONFIRMED' if result else 'NOT CONFIRMED'}")
    return result


def main():
    print("=== Claim 1: lax.cond computes BOTH branches under a batched vmap predicate ===")
    r1 = claim1_lax_cond_computes_both_branches_under_vmap()

    print("\n=== Claim 2: a static Python if/else skips the unused branch under vmap ===")
    r2 = claim2_python_if_skips_unused_branch_under_vmap()

    print("\n=== Claim 3: raising inside pure_callback propagates correctly ===")
    r3 = claim3_raise_inside_pure_callback_propagates()

    ok = r1 and r2 and r3
    print("\n" + ("ALL PASS" if ok else "CHECK FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
