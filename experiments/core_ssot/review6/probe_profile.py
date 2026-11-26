import sys, cProfile, pstats, io
sys.argv = ['x', '40']
src = open('probe_construct.py').read().split('# proxy for per-call')[0]
src = src.replace("t0 = time.perf_counter()\ng2 = Graph(", "pr = cProfile.Profile(); pr.enable()\nt0 = time.perf_counter()\ng2 = Graph(")
src = src.replace("tot2 = time.perf_counter() - t0", "tot2 = time.perf_counter() - t0; pr.disable()")
exec(compile(src, 'probe', 'exec'))
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(12); print(s.getvalue()[:3500])
