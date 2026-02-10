import numpy as np
import pandas as pd
from scipy.stats import expon
from scipy.ndimage import shift

def level_heights_next_tree(levels_heights):
    n = len(levels_heights)
    level_spans = (levels_heights - shift(levels_heights, 1))[1:]
    live_lineages = np.arange(n, 1 ,-1)
    br_len_levels = level_spans * live_lineages
    sampled_level_idx = np.random.choice(np.arange(0, n-1), p=br_len_levels/br_len_levels.sum())
    rec_pos = level_spans[sampled_level_idx] * np.random.random()
    rec_height = levels_heights[sampled_level_idx] + rec_pos
    coal_spans = [levels_heights[sampled_level_idx] + rec_pos] + level_spans[sampled_level_idx+1:].tolist() #+ [np.inf]

    N = 100

    coal_time = None
    height = 0
    for s, k in zip(coal_spans[:-1], live_lineages[sampled_level_idx:-1]):
        if k == 2:
            break
        rate = (k*(k-1)/2) / (2*N)
        t = expon.rvs(size=1, scale=1/rate)[0]
        # print(t)
        if t < s:
            coal_time = height + t
            break
        height += s
    if coal_time is None:
        rate = 1 / (2*N)
        t = expon.rvs(size=1, scale=1/rate)[0]
        coal_time = height + t
    levels_heights[sampled_level_idx+1] = rec_height + coal_time
    return np.sort(levels_heights)


records = []
levels_heights = np.linspace(0, 10, num=5, dtype=float)
for _ in range(10000):
    levels_heights = level_heights_next_tree(levels_heights)

    # print(''.join([f'{x:>9.3f}' for x in levels_heights]).lstrip())
    records.append(levels_heights.copy())
print(pd.DataFrame(records))