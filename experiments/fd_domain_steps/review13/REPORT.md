# Plan A v2.8 review 13 (Opus, HEAD 1641267b) — SIGN-OFF (no BLOCKER, no MAJOR; 3 MINOR; no probes)

Part A: review 12 resolved: no repair, no saved primal (§4.5 l.342-357; §9 l.611-612; §4.3 l.233-236; Batch 2 l.542-544, l.531); no leftover repair/primal text outside §11 history; tie definition (l.344-350); D-NZ docs placement (l.528-533, §9 l.607-610, §3 l.172-173); host site refusing case (§5 l.413). D-NZ applied consistently; host site the stated exception.
MINOR 1: per-theta exact-moments fallback: exact Jacobian used only where _exact_ok = jnp.all(jnp.isfinite(_exactJ)) (__init__.py:9061-9065), else FD moments (:9100-9104) -> compare pdf always and moments under ~_exact_ok; say "zero FD contribution" when moments exact.
MINOR 2: genuine zero partials logged as ties -> log "zero FD gradient (forward tie; may be a genuine zero)"; align §9 ("capped or relative") with §4.5 (capped/positive/unit) or name host relative branch; delete stale "(the host callback evaluates no primal)" in §5 row l.413.
MINOR 3: host-site band "abs(theta) < 1e-10" -> capped rule NaN for θ<0 -> "0 <= theta < 1e-10".
Sound: no traced selection on undeclared non-refusing slots; no extra forward; NaN comment at __init__.py:9080-9082.
