"""Pin records under the capped step: actual old/new pdf(1..5) values (hex-different entries only)."""
import sys, warnings, json
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug4/tests/pytest'); sys.path.insert(0, '/Users/kmt/phasic-bug4/experiments/bug4_discrete_rowsum/review3')
import _discretize_pins as pins
import importlib.util
src = open('/Users/kmt/phasic-bug4/experiments/bug4_discrete_rowsum/review3/t12.py').read().split("VARS = ")[0]
exec(src)
REC = json.load(open(pins.RECORDS))
tot = {v: 0 for v in ('capA', 'capB', 'capM')}; worst_abs = {v: 0.0 for v in tot}; worst_rel_big = {v: 0.0 for v in tot}
for name, (build, call, theta) in pins.CASES.items():
    if name in pins.EXPECTED_ERRORS: continue
    d = call(build())
    if theta is not None: d.update_weights(theta[-1] if isinstance(theta[0], list) else theta)
    n, E, ab = extract(d); base = run(n, E, ab, 5, 'today')[0]
    pinned = REC['cases'][name]['pdf']
    line = f'{name:26s} pinned==today-replica: {[float(x).hex() for x in pinned] == [float(x).hex() for x in base]}\n     today {[f"{x:.17g}" for x in base]}'
    for v in tot:
        got = run(n, E, ab, 5, v)[0]; diff = [i for i in range(5) if float(got[i]).hex() != float(base[i]).hex()]
        if diff:
            tot[v] += 1; worst_abs[v] = max(worst_abs[v], float(np.abs(got - base).max()))
            big = [abs(got[i] - base[i]) / abs(base[i]) for i in diff if abs(base[i]) > 1e-12]
            if big: worst_rel_big[v] = max(worst_rel_big[v], max(big))
            line += f'\n     {v}  ' + ', '.join(f'pdf({i+1}): {base[i]:.17g} -> {got[i]:.17g}' for i in diff)
    print(line)
print('records changed (of 21):', tot); print('worst ABS change:', worst_abs); print('worst REL change among entries with |old| > 1e-12:', worst_rel_big)
