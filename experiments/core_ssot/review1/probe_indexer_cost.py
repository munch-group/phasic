import time, numpy as np
from phasic import StateIndexer, Property
idx = StateIndexer(lineage=[Property('descendants', max_value=20), Property('pop', max_value=1)])
ps = idx.lineage
N=20000
t=time.perf_counter()
for i in range(N):
    ps.props_to_index(descendants=i%20+1, pop=i%2)
dt=(time.perf_counter()-t)/N
t=time.perf_counter()
for i in range(N):
    ps.index_to_props(i%40)
dt2=(time.perf_counter()-t)/N
print(f"props_to_index {dt*1e6:.2f} us/call; index_to_props {dt2*1e6:.2f} us/call; state_length {idx.state_length}")
