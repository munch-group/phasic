# Refactor plan v2.9 review 14 (Opus) — CHANGES REQUIRED (1 MAJOR, 6 MINOR; no probes)

Part A: review 13 resolved (oracle :572, §9 :887-890; compile fix in 0d :214-224, stop row :598; validator hazard phasic.c:4648-4653 confirmed; in-tree bit :179-184 (phasiccpp.cpp:78-86); field rules :185-189; tape getter :204-210 (phasic_pybind.cpp:1593-1615); D19 baseline :225-232; native DPH :210-213; 0c size :571; OI-118 home :655; stale refs fixed; Plan A interface consistent fd-domain-steps-plan.md:280-287 vs :461-480).
MAJOR 1 (introduced by v2.9): phase 0c (:571) and R7 (:248-259) still require load_cpp_builder to work (source rule for both routes; "both routes work on a non-editable install"; changed tree recompiles; OI-116 wrapper template regenerated) but the compile fix is in 0d and the route doesn't compile at HEAD (phasic_pybind.cpp:5192-5200; review12/p_today_compile.log) -> 0c fails its gate. Fix: 0c applies source rule, cache key, secure dir to pmf_from_cpp only; load_cpp_builder parts of OI-089 and OI-116 move to 0d; update R7 :128-129, :248-259, §8, OPEN-ITEMS.md:114-115.
MINORS
1 D19 baseline "today's route with only the compile fixed, after OI-116" never built -> throwaway worktree build or bare build_model time inside the JIT copy (cleaner).
2 oracle walk omits formula tape -> add ops and constant bits, or evaluate one theta on both and compare bitwise.
3 JIT .so dlopen RTLD_LOCAL (phasic_pybind.cpp:5242) exports every ptd_*/phasic:: symbol; RTLD_GLOBAL host could interpose (guess) -> -fvisibility=hidden exporting only entry/free/test-walk (and/or -Wl,-Bsymbolic).
4 native-DPH C++ model is_discrete False -> Graph.pdf continuous: silently wrong (not regression) -> name remedy (g.is_discrete = True) and offer additive discrete= kw on load_cpp_builder (mirroring pmf_from_cpp) in approval list.
5 refusal axes differ (refactor probes {-1e-12,-1e-9,-5e-8,-1e-7,0} :465 vs Plan A slot values {1,5e-8,0,1-5e-8} fd-domain-steps-plan.md:458-465) -> agreement on overlapping cells (weight mode × kind × rewards × builder -> same refuse/accept).
6 status header; OPEN-ITEMS OI-118 row (:117) should say fixed in 0c.
Nothing else MAJOR: hand-off peak one graph + buffer (:233-236); oracle on test models only; order unchanged.
