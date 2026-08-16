# What each inference path actually computes (source-grounded map)

Established 2026-08-16 by reading source, after the user rightly
objected that this should not have been a question put to them. Every
claim below carries a file:line in the underlying research; the load-
bearing ones are restated here with their citations.

## The headline

**The exact-gradient programme made exact the one quantity the default
likelihood does not use, and left finite differences on the quantity it
does use.**

## 1. The default SVGD fit (leaf 5: continuous times, no rewards)

Selected by the if/elif chain in Graph.svgd (src/phasic/__init__.py
:6537-6544). Note svgd_config.py is a VALIDATOR, not a router — it
raises on bad combinations; selection is the plain chain above.

The objective is SVGD._log_prob_unified (src/phasic/svgd.py:6161):

    log p(theta|D) = sum_i log(v_i(theta) + 1e-10) + log prior
                     [ - lambda * ||m(theta) - m_hat||^2 ]   <-- only if
                                                    regularization > 0

and for this leaf v_i(theta) is the continuous PDF at the observed
time, g.pdf(t_i, granularity=0)
(src/cpp/parameterized/graph_builder.cpp:890).

So the likelihood needs the PDF and nothing else. With the default
regularization=0.0, sample_moments is None (svgd.py:5752) and the
moment-penalty branch (svgd.py:6247) is never taken.

Three consequences, all measured or read from source:

- Moments of order 1 and 2 are nevertheless computed on EVERY forward,
  unconditionally (graph_builder.cpp:900), each costing a chained
  expected_waiting_time solve (:512-552). There is no flag to skip
  them.
- The PDF's gradient is finite differences, always
  (__init__.py:8345-8346, "keeping FD for pmf" at :8350-8351).
- The exact moments Jacobian is computed on every backward and
  contracted against a ZERO cotangent. Confirmed by counting
  pure_callback primitives in jax.make_jaxpr(jax.grad(loss)): six with
  exact_moment_grad=True versus five with it False. On this
  configuration the entire B3 moments machinery buys nothing and costs
  one extra host callback per gradient step.

Moments are not entirely idle — but their real use is at SETUP, not in
the loss: the default prior runs method_of_moments end-to-end
(svgd.py:1400-1411), and the default preconditioner builds a
finite-difference moment Jacobian for kernel scaling (svgd.py
:3505-3562). Both run once.

## 2. The other paths

Leaf 1, epoch / daisy chain (__init__.py:6413 -> :4254): the likelihood
is a joint probability after an n-epoch chain — per-epoch stop
probabilities, with the final epoch read as expected sojourn by default.
Moments are structurally absent (jnp.zeros(2), :4934). Gradient is FD
across all epochs; exact_final_grad=True makes only the final epoch's
slots exact, and Graph.epoch_model does not expose it at all, so
FreeEpochModel.fit is permanently and entirely FD.

Leaf 2, joint-index (__init__.py:6469 -> :8388): the likelihood is a
normalised expected sojourn time, E[soj(v_i)] divided by the terminal
sum (:9074). Moments structurally absent. Exact gradient exists but is
opt-in and off by default.

Leaves 3 and 4, rewards: PDF per feature plus moments; same hybrid as
leaf 5 (PDF finite differences, moments exact).

Graph.mcmc: gradient-free entirely. There is no jax.grad anywhere in
mcmc.py; the model's custom_vjp is dead code on that path. It still
computes moments 1..2 on every likelihood evaluation and discards them
(__init__.py:6784-6794, mcmc.py:467).

Graph.method_of_moments: residuals on raw moments, minimised by
scipy.least_squares with no jac= argument (method_of_moments.py
:473-478), so scipy computes its own 2-point forward differences. The
model's custom_vjp never fires.

Graph.probability_matching: residuals on normalised expected sojourn
times; again scipy least_squares with no jac= (:281-286).

model_selection.likelihood_ratio_test: pure forward by default; only
refine=True takes gradients.

bffg: gradient-free by construction — compute_sojourn_times_ffi has no
VJP at all, and jax.grad through it raises.

## 3. The gradient coverage that actually exists

There are exactly two exact-adjoint families in the C layer: the moment
vector (reverse-mode, ptd_moments_grad_theta and variants, sharing
ptd_b3_moments_core at src/c/phasic.c:10971) and the expected-sojourn
vector (forward-mode, ptd_sojourn_grad_theta_subset at :11956).

**There is no PDF or PMF gradient anywhere — not in the FFI, not in
pybind, not in C.** No FFI handler computes any gradient; all nine
registered targets are forward-only. Every exact gradient reaches JAX
through a pure_callback into pybind, assembled Python-side.

## 4. Why this matters for planning

Mapping the coverage onto the paths:

- The default SVGD fit depends on the PDF. Exact coverage of the PDF:
  zero, everywhere in the library. This is precisely what Deferred 3
  would build — and Deferred 3 was the unit I had been treating as the
  most optional of the three.
- The joint-prob and epoch fits depend on sojourn times and joint
  probabilities. Exact coverage exists (Batches E/F/H) but is opt-in,
  and is capped at roughly 8,000 vertices by the tape size guard.
- Moments are exact and default-on, but only enter a likelihood when
  the user sets regularization > 0.

So the programme's default-on exactness sits on a quantity that is off
the critical path for the common case, while the quantity on the
critical path has no exact gradient at all. That is not a criticism of
any individual batch — each was scoped, reviewed and gated correctly
against its own goal — but the goals were never checked against which
quantity the likelihood actually consumes.

## 5. Documentation defects found while establishing this

The atlas set (atlas/*.md plus exact-fd-atlas-SUMMARY.md) is a
2026-08-04 snapshot predating Batches A, B, C, D.4, E, F, G.1, G.2 and
H. It carries no staleness banner, yet memory instructs "read before
any B3 batch". It is wrong in at least six places: it claims svgd has
no exact-vs-FD knob (there are three, __init__.py:5580-5582); that the
daisy chain has no exact implementation (Batch H shipped one); that
baked/dedup mode is statically excluded (Batch E supports it); that the
1-D rewards guard fires on every step (Batch A threads rewards into the
exact path); that the multivariate wrapper has no exact_moment_grad
kwarg (it does, :9220); and that moments_from_graph breaks under vmap
(fixed in Batch D Tier-1).

CLAUDE.md contradicts itself on the multivariate kwarg — it records the
Batch G.2 fix in one place and the pre-fix gap in another.

Two dead kwargs: pmf_from_graph(use_cache=...) is never read; and
moments_from_graph(use_ffi=...) selects no FFI path, only skipping x64
enablement.

Graph.svgd's docstring tells users to build models with pmf_from_graph
or pmf_from_cpp; SVGD.__init__ rejects exactly those, because they do
not return moments (svgd.py:5793-5799).
