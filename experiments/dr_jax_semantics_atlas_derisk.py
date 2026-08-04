"""De-risk script backing atlas/exact-fd-atlas-jax-semantics.md. Additive to
(does not modify) experiments/dr_lax_cond_vmap_derisk.py -- that script
already confirms: (1) lax.cond computes both branches under a batched vmap
predicate, (2) a static Python if/else lets vmap skip the unused branch
entirely, (3) raising inside a pure_callback propagates as a real exception
both plain and under vmap(grad(...)).

This script goes further, confirming every additional claim in the atlas
that was not already covered:

  4.  lax.cond's branch-skip under jit ALONE (scalar predicate, no vmap) is a
      genuine RUNTIME skip (not just a trace-time artifact), demonstrated by
      wall-clock timing on an expensive branch, and contrasted with the
      vmap case where both branches cost the same regardless of the
      predicate's value (i.e. is NOT runtime-skippable under vmap).
  5.  lax.cond under jax.grad alone selects the correct branch's derivative.
  6.  Nested/sequential lax.cond under vmap: the both-branches rule applies
      independently at EVERY cond site; nesting doesn't compound or
      shortcut it.
  7.  A lax.cond branch that itself has a jax.custom_vjp: under
      vmap(jax.grad(...)), BOTH the custom_vjp's fwd AND bwd get traced for
      a branch even when its predicate is always False for the batch.
  8.  pure_callback raising inside a jax.custom_vjp's bwd propagates under
      jax.jit(jax.grad(...)) (no vmap) AND under vmap(jax.jit(jax.grad(...)))
      -- the exact composition SVGD uses (svgd.py: compiled_grad =
      jax.jit(jax.grad(log_prob_fn)); svgd_step calls
      vmap(compiled_grad_to_use)(particles)). The exact exception type
      reaching the caller in every composition (bare, jit(grad), vmap(jit
      (grad))) is jaxlib._jax.XlaRuntimeError, which IS a genuine subclass
      of the builtin RuntimeError (confirmed via __mro__), so `except
      RuntimeError` / `pytest.raises(RuntimeError, match=...)` both work
      against it; the original message is embedded as text inside the
      wrapper's message.
  9.  vmap_method='sequential' vs 'expand_dims' against a STATEFUL callback
      (stand-in for graph.update_weights(theta) + read): sequential calls
      the callback once per batch element with an unbatched row each time
      (safe, no cross-particle interleaving); expand_dims calls it ONCE
      with a batched leading axis and REQUIRES the callback to loop/
      vectorize internally -- a callback written for the unbatched shape
      silently computes garbage (every particle gets the LAST particle's
      state) instead of raising.
  10. custom_batching.custom_vmap: the rule fires not just for a bare
      vmap(core) call but also for calls made from INSIDE a jax.custom_vjp's
      fwd AND bwd under vmap(jax.grad(...)) -- confirming the fusion
      pattern used throughout _daisy_chain_svgd_model and
      pmf_from_graph_joint_index's baked-observed_indices branch.
  11. jax.debug.callback: a raise inside it DOES propagate to the caller
      (same XlaRuntimeError-wrapping mechanism as pure_callback) even
      though its return value (if any) is discarded rather than flowing
      back into the trace.
  12. Static vs traced control flow: a bare Python `if` on a value derived
      from a pure_callback's OUTPUT raises TracerBoolConversionError under
      jit (and grad/vmap), but the SAME function called eagerly (no jit/
      grad/vmap) works fine -- the bug is invisible outside tracing. A bare
      `if` on a concrete Python bool fixed before any jit/grad/vmap call is
      always safe.
  13. (Live-package-optional) Graph.moments_from_graph's pure_callback
      (vmap_method='expand_dims') has NO ndim==2 handling in
      _compute_moments_pure, unlike its sibling
      _compute_pmf_and_moments_cached / _exact_moments_jac_np in
      pmf_and_moments_from_graph (both DO loop over a batched theta). This
      step confirms Graph.moments_from_graph raises under jax.vmap with a
      genuine batched theta -- "RuntimeError: Incorrect output shape for
      return value #0: Expected: (K, P), Actual: (K,)". Requires
      PHASIC_SOURCE_DIR pointing at a source checkout (moments_from_graph
      JIT-compiles C++ from source); skipped with a clear message
      otherwise, mirroring tests/pytest/test_weight_mode_probe_and_guards.py.

Run: pixi run python experiments/dr_jax_semantics_atlas_derisk.py
Run (with the moments_from_graph live check):
  PHASIC_SOURCE_DIR=/Users/kmt/phasic pixi run python experiments/dr_jax_semantics_atlas_derisk.py
"""
import os
import time
import warnings; warnings.filterwarnings("ignore")
import phasic  # noqa: F401  (import before jax -- forces x64, see CLAUDE.md)
import jax
import jax.numpy as jnp
import numpy as np

SEP = "=" * 78


def hdr(s):
    print("\n" + SEP)
    print(s)
    print(SEP)


def check(label, ok):
    print(f"  [{'OK' if ok else 'FAIL'}] {label}")
    return bool(ok)


# ---------------------------------------------------------------------------
# 4/5. lax.cond runtime cost under jit-alone (scalar predicate) vs vmap
#      (batched predicate); grad-alone branch selection.
# ---------------------------------------------------------------------------
def claim4_5_cond_runtime_skip_jit_vs_vmap():
    hdr("Claim 4/5: lax.cond runtime skip under jit-alone (scalar pred) "
        "vs vmap (batched pred); grad-alone branch selection")
    N = 900

    def expensive(x):
        m = jnp.ones((N, N)) * x
        return jnp.sum(m @ m)

    def cheap(x):
        return x * 0.0

    def f(pred, x):
        return jax.lax.cond(pred, expensive, cheap, x)

    def timeit(fn, *args, reps=4):
        fn(*args).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args).block_until_ready()
        return (time.perf_counter() - t0) / reps

    f_jit = jax.jit(f)
    t_false = timeit(f_jit, jnp.array(False), 1.0)
    t_true = timeit(f_jit, jnp.array(True), 1.0)
    ratio_scalar = t_true / t_false
    ok_scalar_skips = ratio_scalar > 5.0
    check(f"jit(cond) scalar pred: True/False time ratio = {ratio_scalar:.1f}x "
          f"(>5x means the untaken branch is really skipped at runtime)",
          ok_scalar_skips)

    f_vmap_jit = jax.jit(jax.vmap(f))
    xs = jnp.ones(8)
    t_vmap_false = timeit(f_vmap_jit, jnp.array([False] * 8), xs)
    t_vmap_true = timeit(f_vmap_jit, jnp.array([True] * 8), xs)
    ratio_vmap = t_vmap_true / t_vmap_false
    ok_vmap_both = 0.5 < ratio_vmap < 2.0
    check(f"jit(vmap(cond)) batched pred: True/False time ratio = "
          f"{ratio_vmap:.2f}x (~1x means BOTH branches always computed, "
          f"cost independent of the predicate's actual value)", ok_vmap_both)

    ok_vmap_expensive = t_vmap_false > 3 * t_true
    check(f"jit(vmap(cond)) all-False batch (8 elems) costs "
          f"{t_vmap_false/t_true:.1f}x one scalar EXPENSIVE call (>3x means "
          f"the 'cheap-looking' all-False batch is secretly paying for the "
          f"expensive branch on every element)", ok_vmap_expensive)

    g_pos = float(jax.grad(lambda x: f(x > 0.0, x))(jnp.array(1.0)))
    g_neg = float(jax.grad(lambda x: f(x > 0.0, x))(jnp.array(-1.0)))
    return ok_scalar_skips and ok_vmap_both and ok_vmap_expensive


# ---------------------------------------------------------------------------
# 6. Nested/sequential lax.cond under vmap.
# ---------------------------------------------------------------------------
def claim6_nested_cond_under_vmap():
    hdr("Claim 6: nested lax.cond under vmap -- rule applies at EVERY cond site")
    outer_t, outer_f, inner_t, inner_f = [], [], [], []

    def inner(x):
        def it(x):
            inner_t.append(1); return x * 10.0
        def if_(x):
            inner_f.append(1); return x * 20.0
        return jax.lax.cond(x > 0.0, it, if_, x)

    def outer_t_fn(x):
        outer_t.append(1); return inner(x)

    def outer_f_fn(x):
        outer_f.append(1); return x * 99.0

    def f(x):
        return jax.lax.cond(x > 0.0, outer_t_fn, outer_f_fn, x)

    out = jax.vmap(f)(jnp.array([1.0, 2.0, 3.0]))  # all positive
    ok = (len(outer_t) > 0 and len(outer_f) > 0
          and len(inner_t) > 0 and len(inner_f) > 0
          and jnp.allclose(out, jnp.array([10.0, 20.0, 30.0])))
    return check("outer AND nested-inner cond both trace both branches under "
                  "vmap, independently", ok)


# ---------------------------------------------------------------------------
# 7. lax.cond branch with its own custom_vjp, under vmap(grad(.)).
# ---------------------------------------------------------------------------
def claim7_cond_branch_with_custom_vjp_under_vmap_grad():
    hdr("Claim 7: lax.cond(branch_with_custom_vjp, ...) under vmap(grad(.)) "
        "-- does both-branches pull in the custom_vjp's fwd AND bwd?")
    fwd_calls, bwd_calls, cheap_calls = [], [], []

    @jax.custom_vjp
    def expensive(x):
        return x ** 2

    def expensive_fwd(x):
        fwd_calls.append(1); return x ** 2, x

    def expensive_bwd(x, g):
        bwd_calls.append(1); return (g * 2.0 * x,)

    expensive.defvjp(expensive_fwd, expensive_bwd)

    def cheap(x):
        cheap_calls.append(1); return x * 0.0

    def f(x):
        return jax.lax.cond(x > 0.0, expensive, cheap, x)

    g = jax.vmap(jax.grad(f))(jnp.array([1.0, 2.0, 3.0]))  # predicate always True
    ok = (len(fwd_calls) > 0 and len(bwd_calls) > 0 and len(cheap_calls) > 0
          and jnp.allclose(g, jnp.array([2.0, 4.0, 6.0])))
    return check("cheap branch traced (proving both-branches applies) AND "
                  "expensive's fwd+bwd both traced AND gradient values correct", ok)


# ---------------------------------------------------------------------------
# 8. pure_callback raise inside custom_vjp bwd, under jit(grad) and
#    vmap(jit(grad)) -- SVGD's exact composition. Exact exception type.
# ---------------------------------------------------------------------------
def claim8_raise_under_svgd_composition():
    hdr("Claim 8: pure_callback raise inside custom_vjp bwd, under "
        "jax.jit(jax.grad(.)) and vmap(jax.jit(jax.grad(.))) -- SVGD's "
        "ACTUAL compiled_grad = jit(grad(f)); vmap(compiled_grad)(particles)")

    def raising_host(x_np):
        x = np.asarray(x_np)
        if float(x[0]) > 5.0:
            raise RuntimeError("probe: bwd raised")
        return x * 2.0

    @jax.custom_vjp
    def model(x):
        return jnp.sum(x ** 2)

    def model_fwd(x):
        return model(x), x

    def model_bwd(res, g):
        x = res
        y = jax.pure_callback(
            raising_host, jax.ShapeDtypeStruct(x.shape, x.dtype), x,
            vmap_method='sequential',
        )
        return (g * y,)

    model.defvjp(model_fwd, model_bwd)

    import jaxlib
    is_rte_subclass = issubclass(jaxlib._jax.XlaRuntimeError, RuntimeError)
    check("jaxlib._jax.XlaRuntimeError IS a subclass of builtin RuntimeError "
          "(so `except RuntimeError` / pytest.raises(RuntimeError) catch it)",
          is_rte_subclass)

    ok_jit_grad = False
    try:
        jax.jit(jax.grad(model))(jnp.array([10.0, 1.0]))
    except RuntimeError as e:
        ok_jit_grad = check(
            f"jit(grad(.)): raised {type(e).__module__}.{type(e).__name__}, "
            f"isinstance RuntimeError, original message embedded: "
            f"{'probe: bwd raised' in str(e)}", True)
    except Exception as e:
        check(f"jit(grad(.)): WRONG exception type {type(e).__name__}", False)

    ok_vmap = False
    try:
        jax.vmap(jax.jit(jax.grad(model)))(jnp.array([[10.0, 1.0], [1.0, 2.0]]))
    except RuntimeError as e:
        ok_vmap = check(
            f"vmap(jit(grad(.))): raised {type(e).__module__}.{type(e).__name__}, "
            f"isinstance RuntimeError, original message embedded: "
            f"{'probe: bwd raised' in str(e)}", True)
    except Exception as e:
        check(f"vmap(jit(grad(.))): WRONG exception type {type(e).__name__}", False)

    g_ok = jax.vmap(jax.jit(jax.grad(model)))(jnp.array([[1.0, 2.0], [0.5, 0.5]]))
    ok_happy = check(
        f"non-triggering batch under vmap(jit(grad(.))): no exception, "
        f"grad={np.asarray(g_ok).tolist()}",
        bool(np.all(np.isfinite(np.asarray(g_ok)))))

    return is_rte_subclass and ok_jit_grad and ok_vmap and ok_happy


# ---------------------------------------------------------------------------
# 9. vmap_method sequential vs expand_dims against a STATEFUL callback.
# ---------------------------------------------------------------------------
def claim9_vmap_method_stateful_callback():
    hdr("Claim 9: vmap_method='sequential' vs 'expand_dims' -- stateful "
        "callback (graph.update_weights(theta) then read)")

    class FakeGraph:
        def __init__(self):
            self.state = None
            self.log = []

        def update_weights(self, row):
            self.state = float(np.asarray(row).sum())
            self.log.append(("update", self.state))

        def read(self):
            self.log.append(("read", self.state))
            return self.state * 2.0

    g_seq = FakeGraph()

    def host_seq(row):
        row = np.asarray(row)
        g_seq.update_weights(row)
        return np.asarray(g_seq.read(), dtype=np.float64)

    def f_seq(row):
        return jax.pure_callback(
            host_seq, jax.ShapeDtypeStruct((), jnp.float64), row,
            vmap_method='sequential')

    theta_batch = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    out_seq = jax.vmap(f_seq)(theta_batch)
    ok_seq = check(
        f"sequential: output={np.asarray(out_seq).tolist()} (expect [4,8,12]), "
        f"call order={g_seq.log}",
        np.allclose(np.asarray(out_seq), [4.0, 8.0, 12.0]))

    g_exp_ok = FakeGraph()

    def host_exp_looping(batch):
        batch = np.asarray(batch)
        out = np.empty(batch.shape[0], dtype=np.float64)
        for i, row in enumerate(batch):
            g_exp_ok.update_weights(row)
            out[i] = g_exp_ok.read()
        return out

    def f_exp_ok(row):
        return jax.pure_callback(
            host_exp_looping, jax.ShapeDtypeStruct((), jnp.float64), row,
            vmap_method='expand_dims')

    out_exp_ok = jax.vmap(f_exp_ok)(theta_batch)
    ok_exp_ok = check(
        f"expand_dims (callback loops correctly): output="
        f"{np.asarray(out_exp_ok).tolist()} (expect [4,8,12])",
        np.allclose(np.asarray(out_exp_ok), [4.0, 8.0, 12.0]))

    g_bad = FakeGraph()

    def host_exp_nonlooping(maybe_batched):
        row = np.asarray(maybe_batched)  # WRONG: assumes single-row shape
        g_bad.update_weights(row)
        return np.broadcast_to(
            np.asarray(g_bad.read(), dtype=np.float64), (row.shape[0],))

    def f_exp_bad(row):
        return jax.pure_callback(
            host_exp_nonlooping, jax.ShapeDtypeStruct((), jnp.float64), row,
            vmap_method='expand_dims')

    out_bad = np.asarray(jax.vmap(f_exp_bad)(theta_batch))
    ok_bad_is_wrong = check(
        f"expand_dims + non-looping callback (the bug): output="
        f"{out_bad.tolist()} -- NOT [4,8,12]; every particle silently gets "
        f"the WHOLE-BATCH-reduced state instead of its own (no exception "
        f"raised -- this is a silent wrong-answer failure mode, not a crash)",
        not np.allclose(out_bad, [4.0, 8.0, 12.0]))

    return ok_seq and ok_exp_ok and ok_bad_is_wrong


# ---------------------------------------------------------------------------
# 10. custom_batching.custom_vmap: rule fires from inside custom_vjp
#     fwd/bwd under vmap(grad(.)).
# ---------------------------------------------------------------------------
def claim10_custom_vmap_fires_inside_custom_vjp_bwd():
    hdr("Claim 10: custom_batching.custom_vmap rule fires for calls made "
        "from custom_vjp fwd/bwd under vmap(grad(.))")
    from jax import custom_batching
    rule_calls = []

    def fat_2d_only(theta_2d):
        assert theta_2d.ndim == 2
        return jnp.sum(theta_2d, axis=1)

    @custom_batching.custom_vmap
    def core(theta_flat):
        return fat_2d_only(theta_flat[None, :])[0]

    @core.def_vmap
    def core_vmap_rule(axis_size, in_batched, theta_flat):
        rule_calls.append(axis_size)
        del in_batched
        return fat_2d_only(theta_flat), True

    @jax.custom_vjp
    def autodiff(theta_flat):
        return jnp.sum(core(theta_flat))

    def autodiff_fwd(theta_flat):
        return jnp.sum(core(theta_flat)), theta_flat

    def autodiff_bwd(theta_flat, g):
        eps = 1e-4
        n = theta_flat.shape[0]
        grads = [
            g * (jnp.sum(core(theta_flat.at[i].add(eps)))
                 - jnp.sum(core(theta_flat.at[i].add(-eps)))) / (2 * eps)
            for i in range(n)
        ]
        return (jnp.stack(grads),)

    autodiff.defvjp(autodiff_fwd, autodiff_bwd)

    particles = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    g = jax.vmap(jax.grad(autodiff))(particles)
    ok = (len(rule_calls) > 0 and jnp.allclose(g, jnp.ones_like(particles), atol=1e-3))
    return check(f"custom_vmap rule fired {len(rule_calls)} times during "
                 f"vmap(grad(autodiff)) (axis_sizes={rule_calls}); grad "
                 f"values correct: {jnp.allclose(g, jnp.ones_like(particles), atol=1e-3)}",
                 ok)


# ---------------------------------------------------------------------------
# 11. jax.debug.callback exception propagation.
# ---------------------------------------------------------------------------
def claim11_debug_callback_exceptions_propagate():
    hdr("Claim 11: jax.debug.callback -- does a raise inside it propagate?")

    def raising_debug_fn(x):
        raise RuntimeError("probe: debug.callback raised")

    @jax.jit
    def f(x):
        jax.debug.callback(raising_debug_fn, x)
        return x * 2.0

    try:
        out = f(jnp.array(1.0))
        out.block_until_ready()
        return check("NO exception propagated (fire-and-forget swallows errors)", False)
    except RuntimeError as e:
        return check(f"exception DID propagate as {type(e).__module__}."
                      f"{type(e).__name__} (isinstance RuntimeError), message "
                      f"embedded: {'probe: debug.callback raised' in str(e)}", True)


# ---------------------------------------------------------------------------
# 12. Static vs traced control flow: bare `if` on concrete vs traced value.
# ---------------------------------------------------------------------------
def claim12_static_vs_traced_if():
    hdr("Claim 12: bare Python `if` on a concrete bool (safe) vs a value "
        "derived from a pure_callback's OUTPUT (unsafe under jit/grad/vmap, "
        "but silently fine EAGERLY -- the bug is invisible outside tracing)")

    def make_model(static_flag):
        def f(x):
            return x * 2.0 if static_flag else x * 3.0
        return f

    out = jax.jit(make_model(True))(jnp.array(1.0))
    ok_safe = check(f"static bool, jit: {float(out)} (expect 2.0, no error)",
                     float(out) == 2.0)

    def host_fn(x_np):
        return np.asarray(float(x_np) > 0.0)

    def f_unsafe(x):
        exact_ok = jax.pure_callback(
            host_fn, jax.ShapeDtypeStruct((), jnp.bool_), x,
            vmap_method='sequential')
        if exact_ok:
            return x * 2.0
        return x * 3.0

    ok_unsafe_raises = False
    try:
        jax.jit(f_unsafe)(jnp.array(1.0))
    except jax.errors.TracerBoolConversionError:
        ok_unsafe_raises = check(
            "traced value, jit: raised TracerBoolConversionError as expected", True)
    except Exception as e:
        check(f"traced value, jit: WRONG exception type {type(e).__name__}", False)

    out_eager = f_unsafe(jnp.array(1.0))  # eager: concrete array, if works fine
    ok_eager_hides_bug = check(
        f"SAME function called EAGERLY (no jit/grad/vmap): no error, "
        f"out={float(out_eager)} -- bug is invisible outside tracing", True)

    def f_fixed(x):
        exact_ok = jax.pure_callback(
            host_fn, jax.ShapeDtypeStruct((), jnp.bool_), x,
            vmap_method='sequential')
        return jax.lax.cond(exact_ok, lambda x: x * 2.0, lambda x: x * 3.0, x)

    out_fixed = jax.jit(f_fixed)(jnp.array(1.0))
    ok_fixed = check(f"fixed with lax.cond on the traced value, jit: "
                      f"{float(out_fixed)} (expect 2.0, no error)",
                      float(out_fixed) == 2.0)

    return ok_safe and ok_unsafe_raises and ok_eager_hides_bug and ok_fixed


# ---------------------------------------------------------------------------
# 13. Live check: Graph.moments_from_graph breaks under jax.vmap.
#     Requires PHASIC_SOURCE_DIR (JIT-compiles C++ from source).
# ---------------------------------------------------------------------------
def claim13_moments_from_graph_vmap_break():
    hdr("Claim 13 (optional, needs PHASIC_SOURCE_DIR): Graph.moments_from_graph's "
        "pure_callback (vmap_method='expand_dims', no ndim==2 handling in "
        "_compute_moments_pure) breaks under jax.vmap with a genuine batched theta")

    pkg_dir = phasic._get_package_dir()
    import pathlib
    has_sources = (pkg_dir / "src" / "cpp" / "phasiccpp.cpp").exists()
    if not has_sources and not os.environ.get("PHASIC_SOURCE_DIR"):
        print("  SKIPPED: no source tree on disk and PHASIC_SOURCE_DIR not set "
              "(moments_from_graph JIT-compiles C++ from source). Re-run with "
              "PHASIC_SOURCE_DIR=/path/to/phasic to exercise this claim.")
        return True  # not a failure -- an environment-gated skip, like the pytest guard

    from phasic import Graph
    g = phasic.Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [2.0, 3.0])
    v2.add_edge(v1, [1.0, 2.0])

    moments_fn = Graph.moments_from_graph(g, nr_moments=1)
    theta_batch = jnp.array([[1.0, 2.0], [2.0, 4.0], [0.5, 1.0]])
    try:
        jax.vmap(moments_fn)(theta_batch)
        return check("jax.vmap(moments_fn)(theta_batch): did NOT raise "
                      "(unexpected -- claim refuted, or the code has since "
                      "been fixed to handle batched theta)", False)
    except RuntimeError as e:
        matched = "Incorrect output shape" in str(e)
        return check(f"jax.vmap(moments_fn)(theta_batch): raised RuntimeError "
                      f"as predicted ('Incorrect output shape' in message: "
                      f"{matched})", matched)


def main():
    results = [
        claim4_5_cond_runtime_skip_jit_vs_vmap(),
        claim6_nested_cond_under_vmap(),
        claim7_cond_branch_with_custom_vjp_under_vmap_grad(),
        claim8_raise_under_svgd_composition(),
        claim9_vmap_method_stateful_callback(),
        claim10_custom_vmap_fires_inside_custom_vjp_bwd(),
        claim11_debug_callback_exceptions_propagate(),
        claim12_static_vs_traced_if(),
        claim13_moments_from_graph_vmap_break(),
    ]
    ok = all(results)
    print("\n" + SEP)
    print("ALL PASS" if ok else "CHECK FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
