"""Does MCMC(parallel='processes') run at all on the unmodified build with a fixed slot, and under the wrapper?"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug2/experiments/bug2_fixed_values/review2')
import numpy as np
def main():
    import jax.numpy as jnp
    import proto
    from phasic import Graph, GaussPrior, MCMC
    d = np.array([0.3, 0.5, 1.0, 1.5, 2.0, 0.8, 1.2, 0.6])
    for patched in (False, True):
        if patched: proto.install('plan')
        try:
            base = Graph.pmf_and_moments_from_graph(proto.graph3(), nr_moments=2, theta_dim=2)
            m = MCMC(model=base, observed_data=d, theta_dim=2, fixed=[(1, 0.3)], n_chains=2, n_samples=20, burn_in=10,
                     progress=False, verbose=False, seed=1, parallel='processes', prior=[GaussPrior(ci=[0.5, 5], lower=0.0), None])
            m.run(); r = m.get_results()
            print(f'patched={patched}: OK fixed column unique={np.unique(np.asarray(r["particles"])[:,1])}')
        except Exception as e:
            print(f'patched={patched}: RAISED {type(e).__name__}: {str(e)[:300]}')
        finally: proto.uninstall()
if __name__ == '__main__':
    main()
