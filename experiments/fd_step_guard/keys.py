import sys; sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed
g, idx = coalescent_indexed(3)
J = g.joint_prob_graph(idx, reward_limit=10, discrete=False)
t = J.joint_prob_table() if hasattr(J, 'joint_prob_table') else None
print(t.columns.tolist() if t is not None else None); print(t.head(12).to_string() if t is not None else '')
