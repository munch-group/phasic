# Refactor plan v2.8 review 13 (Opus) — CHANGES REQUIRED (1 MAJOR, minors; no probes)

Part A: review 12 resolved (0d row :546; R7 :172-209; native checks :503-505; stop row :572; 0c :545; phase 3/5/12 rows :549, :551, :560; OI-118 recorded; routing corpus :457-460). Stale: R7 "0c" at :131, :140, :166; routing gate :464; §6 :614; refusal baseline "after 0c" :443; phase-1 baseline :547 (should be after 0d); OI-118 no §8 home; OPEN-ITEMS.md:114, :119 say 0c.
MAJOR 1: 0d gate self-referential (walk of REBUILT graph + idempotence); JIT graph freed before rebuild -> symmetric field drop passes (is_aux phasic.h:288, parameterized IPV coefficients, was_dph, current edge weights, edge-kind interleaving). Fix: test-only exported function in JIT copy writes canonical C-level walk of SOURCE graph before free (vertex index, state, is_aux, edge target, weight bits, coefficient bits in order, IPV, was_dph, param_length, formula source, theta blocks/names, epoch fields); same walk on rebuilt graph; compare; optionally today's pointer route as A/B oracle in default build during development; each test model asserts expected header values (was_dph -> is_discrete True; formula -> formula).
MINORS
1 0c compile fix re-enables live pointer crossing until 0d; validator build (CMakeLists.txt:23-24) adds _dbg_off_clean (phasic.h:263-270); extension's ptd_graph_destroy reads it (phasic.c:4648-4653) past JIT-allocated struct -> move compile fix into 0d (or reproduce defines and state crossing live until 0d).
2 duplicate states/AVL: phasic::Graph::create_vertex doesn't insert into AVL (phasiccpp.cpp:80-86); rebuild reinserting first-wins (ptd_clone_graph phasic.c:1370-1374) changes find_vertex -> per-vertex "in tree" bit or listed difference; test find_vertex.
3 ptd_graph fields (phasic.h:172-262): edge_mode, param_length_locked, parameterized reconstructed via add-edge path; current_params, weight_version fresh; carry current edge WEIGHT explicitly.
4 tape getter is new (only _set_weight_tape/_clear_weight_tape bound phasic_pybind.cpp:1593-1615; ptd_weight_tape private phasic.c:5106); reproduce compile_formula dict shape; wrapper must not set weight_formula=src (Python recompile may diverge, OI-048).
5 D19 baseline should exclude normalize (removed by OI-116 in 0c).
6 native DPH C++ model -> is_discrete False (only from was_dph); list until phase 4.
7 0c size: M, or S relative to 0d.
8 OI-118 root in copy-assignment (phasiccpp.h:~1840-1900); 0d code avoids Graph assignment until fixed.
Refs: phasic_pybind.cpp:5141-5268; __init__.py:12685-12742; phasiccpp.h:254-282, :1875-1930.
