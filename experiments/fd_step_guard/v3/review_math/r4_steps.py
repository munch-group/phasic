"""R4: predicted steps, shipped (floor(G t)+1, G = max(1000, int(2*max(512,R)))) vs new
(n = L + 10 sqrt(L+1) + 30, L = c R t; c = 1.02 and the prototype's effective c = 2)."""
import math
ts = (1e-3, 1e-2, 1e-1, 1, 10, 100)
print('ratio new/shipped steps (c=1.02) [new steps];  rows: max rate R, cols: t = ' + ' '.join(f'{t:g}' for t in ts))
for R in (1, 10, 45, 100, 512, 1e3, 4950, 1e4):
    row = []
    for t in ts:
        G = max(1000, int(2 * max(512, R))); sh = int(G * t) + 1
        L = 1.02 * R * t; nw = int(L + 10 * math.sqrt(L + 1) + 30) + 1
        row.append(f'{nw / sh:7.2f}[{nw}]')
    print(f'  R={R:6g}: ' + ' '.join(row))
