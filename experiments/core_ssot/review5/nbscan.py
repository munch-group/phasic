import json,glob,re,collections,sys
pats={
 'P2 trace API':r'record_elimination_trace|evaluate_trace|instantiate_from_trace|trace_elimination|compute_trace|cache_trace|hierarchical_trace_cache|trace_cache|_ensure_trace|trace_serialization',
 'P4 moments_from_graph/use_ffi':r'moments_from_graph|use_ffi',
 'P4 moments/variance/cov':r'\.(moments|variance|covariance|expectation)\(',
 'P5 serialize/clone/pickle':r'\.serialize\(|from_serialized|pickle|\.clone\(',
 'P5 graph attrs':r'\.is_discrete|_weight_mode|\.weight_mode|weight_formula|weight_callback|was_dph',
 'P6 StateIndexer':r'StateIndexer|Property\(|PropertySet|\.props\b|state_to_index|index_to_state|\.to_dict\(',
 'P6 graph_cache':r'graph_cache|GraphCache|cache_graph|use_cache',
 'P7 add_epoch':r'add_epoch\(',
 'P8 joint':r'joint_prob_graph|joint_stop_prob_graph|joint_sojourn_graph|joint_prob_table|EpochContext|absorbing_state_rewards',
 'P8 private joint attrs':r'\._(joint_[a-z_]+|indexer|base_indexer|epoch_state_index|n_epochs)\b',
 'P12 user C++':r'load_cpp_builder|pmf_from_cpp|\.cpp[\'"]',
 'P13 sample':r'\.sample\(',
 'P15 profile':r'\.profile\(',
}
res=collections.defaultdict(list)
for f in sorted(glob.glob('docs/pages/**/*.ipynb',recursive=True)):
    try: nb=json.load(open(f))
    except Exception as e: continue
    for i,c in enumerate(nb['cells']):
        if c['cell_type']!='code': continue
        s=''.join(c['source'])
        for k,p in pats.items():
            if re.search(p,s): res[k].append((f.replace('docs/pages/',''),i))
for k in pats:
    v=res[k]; files=collections.Counter(f for f,_ in v)
    print(f"{k}: {len(v)} cells in {len(files)} notebooks")
    if k.startswith(('P2','P8 private','P12','P4 moments_from')): 
        for f,n in files.items(): print('   ',f,[i for ff,i in v if ff==f])
