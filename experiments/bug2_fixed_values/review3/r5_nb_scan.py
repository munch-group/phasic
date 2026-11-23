"""Read-only scan of notebooks/docs for fixed= usages (code cells), with cell index and nearby value definitions."""
import json, re, sys, glob, os
for root in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(root, 'docs/pages/**/*.ipynb'), recursive=True)):
        try: nb = json.load(open(p))
        except Exception as e: print('ERR', p, e); continue
        defs = {}
        for ci, c in enumerate(nb.get('cells', [])):
            if c.get('cell_type') != 'code': continue
            src = ''.join(c.get('source', []))
            for m in re.finditer(r'^\s*(mutation_rate|mut_rate|rec_rate|pop_size|N|migration_rate|\w*rate\w*)\s*=\s*([^\n#]+)', src, re.M):
                defs.setdefault(m.group(1), []).append((ci, m.group(2).strip()[:40]))
            for ln in src.splitlines():
                if re.search(r'\bfixed\s*=', ln) or 'param_transform' in ln or 'positive_params' in ln:
                    print(f'{p.replace(root, os.path.basename(root))} [cell {ci}] {ln.strip()[:150]}')
        used = {k: v for k, v in defs.items() if k in ('mutation_rate', 'mut_rate', 'rec_rate', 'pop_size', 'N')}
        if used: print('    defs:', used)
