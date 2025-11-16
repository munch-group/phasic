import numpy as np
import pandas as pd
from scipy.stats import expon
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import seaborn as sns

%config InlineBackend.figure_format = 'retina'

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

def tot_br_len(levels_heights):
    level_spans = (levels_heights - shift(levels_heights, 1))[1:]
    live_lineages = np.arange(len(levels_heights), 1 ,-1)
    br_len_levels = level_spans * live_lineages
    B = br_len_levels.sum()
    return br_len_levels.sum()


def next_snp_is_new_tree(levels_heights, length):
    prev_B = None
    while True:
        B = tot_br_len(levels_heights)
        if np.random.random() < rec_rate * tot_br_len(levels_heights):
            levels_heights = level_heights_next_tree(levels_heights)
        B = tot_br_len(levels_heights)            
        yield int(B != prev_B)
        prev_B = B

if __name__ == "__main__":
    levels_heights = np.linspace(0, 10, num=5, dtype=float)
    snp_is_new_tree = next_snp_is_new_tree(levels_heights, 1000)
    
    for _ in range(10):
        is_new_tree = next(snp_is_new_tree)
    
        print( next(snp_is_new_tree) )
    
    
    levels_heights = np.linspace(0, 10, num=5, dtype=float)
    records = []
    br_len_list = []
    for i in range(2000):
        B = tot_br_len(levels_heights)
        if np.random.random() < rec_rate * tot_br_len(levels_heights):
            levels_heights = level_heights_next_tree(levels_heights)
        if i < 1000:
            continue
        br_len_list.append(B)
        records.append(levels_heights.copy())
    df = pd.DataFrame(records)
    br_len_list = pd.Series(br_len_list)
    rec_event = br_len_list != br_len_list.shift()
    rec_event = rec_event[1:]


def stairs(df, start='start', end='end', pos='pos', endtrim=0):
    "Turn a df with start, end into one with pos to plot as stairs"
    df1 = df.copy(deep=True)
    df2 = df.copy(deep=True)
    df1[pos] = df1[start]
    df2[pos] = df2[end] - endtrim
    return pd.concat([df1, df2]).sort_values([start, end])
    
plt.figure(figsize=(15, 4))
for col in df:
    _df = df[col].to_frame().reset_index()
    _df['start'] = _df.index.values
    _df['end'] = _df.start + 1
    _df = stairs(_df)
    plt.plot(_df['pos'], _df[col])

# plt.yscale('log')

for x, b in enumerate(rec_event):
    if b:
        plt.vlines([x for (x, b) in enumerate(rec_event) if b], 0, 2000, color='black', linewidth=0.2)
# print(plt.ylim())
sns.despine()
