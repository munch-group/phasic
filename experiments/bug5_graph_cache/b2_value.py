"""What does the graph cache save?  Build time (cache off) vs load time
(cache hit) for the repo's reference model builders. HOME redirected."""
import os, sys, shutil, time
HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, 'fakehome_b2'); shutil.rmtree(FAKE, ignore_errors=True); os.makedirs(FAKE)
os.environ['HOME'] = FAKE
import warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
import phasic
import test_notebook_model_recovery as M
import inspect
def timeit(f):
    t = time.perf_counter(); g = f(); return time.perf_counter() - t, g
for name, mk in (('coalescent n=22', lambda: M.coalescent_graph(22)),
                 ('two_island n=12', lambda: M.two_island_graph(12)),
                 ('two_island n=16', lambda: M.two_island_graph(16)),
                 ('coalescent_indexed n=8', lambda: M.coalescent_indexed(8)[0])):
    phasic.configure(graph_cache=False)
    t_off, g = timeit(mk)
    phasic.configure(graph_cache=True)
    t_miss, _ = timeit(mk)          # builds and saves
    t_hit, g2 = timeit(mk)          # loads
    print(f'{name}: {g.vertices_length()} vertices; build (cache off) {t_off:.3f}s; build+save {t_miss:.3f}s; load (hit) {t_hit:.3f}s; same size: {g2.vertices_length() == g.vertices_length()}')
