"""Review 6: re-measure StateIndexer scalar latency and its share of a
two-island (indexer-based) construction."""
import time, sys
import numpy as np
from phasic import Graph, with_ipv, StateIndexer, Property
n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
ix = StateIndexer(descendants=[Property('pop1', min_value=0, max_value=n),
                               Property('pop2', min_value=0, max_value=n),
                               Property('in_pop', min_value=1, max_value=2)])
d = ix.descendants
K = 20000
t0 = time.perf_counter()
for _ in range(K): d.props_to_index(pop1=1, pop2=0, in_pop=1)
p2i = (time.perf_counter() - t0) / K
t0 = time.perf_counter()
for i in range(K): d.index_to_props(i % d.state_length if hasattr(d,'state_length') else 3)
i2p = (time.perf_counter() - t0) / K
print(f"props_to_index {p2i*1e6:.2f} us, index_to_props {i2p*1e6:.2f} us")
