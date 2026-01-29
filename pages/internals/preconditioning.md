# Kernel Preconditioning for SVGD

## The multi-scale problem

SVGD uses an RBF kernel to measure particle similarity:

$$
K(\theta^{(i)}, \theta^{(l)}) = \exp\!\Bigl(-\sum_j \frac{(\theta^{(i)}_j - \theta^{(l)}_j)^2}{2h_j}\Bigr)
$$

where $h_j$ is the bandwidth in dimension $j$.  The SVGD update for particle $i$ is

$$
\theta^{(i)} \;\leftarrow\; \theta^{(i)} + \epsilon\,\hat\phi(\theta^{(i)}), \qquad
\hat\phi(\theta) = \frac{1}{n}\sum_{l=1}^{n}\bigl[K(\theta^{(l)},\theta)\,\nabla_{\theta^{(l)}}\!\log p(\theta^{(l)}\!\mid\!x) \;+\; \nabla_{\theta^{(l)}}K(\theta^{(l)},\theta)\bigr].
$$

When parameters live on different scales---e.g. $\theta_1 \approx 10$ (a fast coalescence rate) and $\theta_2 \approx 0.01$ (a slow mutation rate)---the kernel treats a unit change in every dimension equally.  Pairwise distances are dominated by the large-scale parameter, so the kernel effectively ignores the small-scale one.  The per-dimension median bandwidth heuristic helps but cannot fully compensate when the scales differ by orders of magnitude.

Preconditioning rescales the particle coordinates before kernel evaluation so that all dimensions contribute comparably to inter-particle distances.

## The preconditioning mechanism

Both preconditioners produce a diagonal scaling vector $\mathbf{D} = (D_1,\ldots,D_p)$, normalized so that $\bar D = 1$.  The kernel is then evaluated in *preconditioned* coordinates

$$
z_j^{(i)} = D_j\,\theta_j^{(i)},
$$

and the kernel gradient is transformed back to the original space by the chain rule:

$$
\frac{\partial K}{\partial \theta_j} = D_j \cdot \frac{\partial K}{\partial z_j}.
$$

A dimension with large $D_j$ is *compressed* in kernel space: particles that are far apart in $\theta_j$ appear closer in $z_j$, making the kernel more sensitive to differences along that direction.  The effect is that the repulsive term $\nabla_\theta K$ pushes particles apart more evenly across all dimensions, preventing collapse onto a lower-dimensional manifold.

Both methods operate identically at the kernel level.  They differ only in how $\mathbf{D}$ is computed.

## Reference point selection

Both preconditioners require evaluating derivatives of the model at a *reference point* $\theta_\text{ref}$.  A poor reference (e.g. the particle mean at initialization, which is drawn from a standard normal prior) can produce meaningless scaling.

The shared reference search (`_find_moment_matching_reference`) performs a coordinate-wise grid search: for each dimension $j$ in turn, it evaluates the model's first moment $\mu_1(\theta)$ over a grid of candidate values and picks the one minimizing $|\mu_1 - \bar x|$, where $\bar x$ is the sample mean of the observed data.  The grid adapts to the data scale:

- Lower end: $\phi = -2$ (corresponding to $\text{softplus}(-2) \approx 0.13$ in constrained space)
- Upper end: $\phi = \text{softplus}^{-1}(10\bar x)$, computed via the numerically stable formula $\log(\text{expm1}(x))$ for $x < 30$ and approximated as $x$ for $x \geq 30$
- Fixed grid: 3 small values $\{-2, -1, 0\}$ plus 12 linearly spaced values from $0.5$ to the upper bound

This produces a reference point in a region where the model is "alive"---where the PMF is non-negligible and the moments are finite---regardless of the prior initialization.

## Moment Jacobian preconditioning (default)

### Mathematical formulation

Let $\mu_k(\theta)$ denote the $k$-th moment of the phase-type distribution parameterized by $\theta$.  The moment Jacobian at the reference point is the matrix

$$
J_{kj} = \frac{\partial \mu_k}{\partial \theta_j}\bigg|_{\theta_\text{ref}}, \qquad k = 1,\ldots,K, \quad j = 1,\ldots,p.
$$

The scaling for dimension $j$ is the $\ell_2$-norm of column $j$:

$$
D_j = \frac{\|\mathbf{J}_{:,j}\|}{\frac{1}{p}\sum_{j'}\|\mathbf{J}_{:,j'}\|}, \qquad
\|\mathbf{J}_{:,j}\| = \sqrt{\sum_k J_{kj}^2}.
$$

(The denominator normalizes $\mathbf{D}$ to have mean 1.)

### Interpretation

$\|\mathbf{J}_{:,j}\|$ measures how much the model's moments respond to changes in $\theta_j$.  A parameter that strongly affects the moments gets a large $D_j$, compressing the kernel in that direction---making the kernel more sensitive to differences along that axis so that particles spread out properly.

### Computation

The Jacobian is estimated by central finite differences with step size $\epsilon = 10^{-5}$:

$$
J_{kj} \approx \frac{\mu_k(\theta_\text{ref} + \epsilon\, e_j) - \mu_k(\theta_\text{ref} - \epsilon\, e_j)}{2\epsilon}.
$$

This requires $2p$ model evaluations (forward pass only, no likelihood division), where $p$ is the number of learnable parameters.

### Why this is the default

1. **Robustness.** The computation involves only moment differences---there is no division by PMF values.  At the reference point the moments are finite by construction (the reference search ensures this), so the Jacobian entries are always well-behaved.

2. **Cost.** Each model evaluation returns both PMF and moments.  The Jacobian method uses only the moments, requiring $2p + 1$ evaluations total ($2p$ for finite differences plus one for the reference moments).

3. **Sufficiency.** For the purpose of kernel preconditioning we need only the *relative* sensitivity of each parameter, not the full curvature of the log-likelihood.  The column norms of the moment Jacobian capture this directly.


## Fisher information preconditioning

### Mathematical formulation

The Fisher information matrix for a model $p(x|\theta)$ is

$$
\mathcal{I}_{jj'} = \mathbb{E}_{x\sim p(\cdot|\theta)}\!\left[\frac{\partial \log p(x|\theta)}{\partial \theta_j}\,\frac{\partial \log p(x|\theta)}{\partial \theta_{j'}}\right].
$$

The preconditioner uses only the diagonal, estimated empirically at the reference point over the observed data $x_1,\ldots,x_N$:

$$
\hat F_j = \frac{1}{N}\sum_{n=1}^{N} s_{nj}^2, \qquad s_{nj} = \frac{\partial \log p(x_n|\theta)}{\partial \theta_j}\bigg|_{\theta_\text{ref}}.
$$

The score $s_{nj}$ is computed from the PMF via

$$
s_{nj} = \frac{1}{p(x_n|\theta_\text{ref})} \cdot \frac{\partial p(x_n|\theta)}{\partial \theta_j}\bigg|_{\theta_\text{ref}},
$$

where the PMF derivative is again estimated by central finite differences.  The scaling is then

$$
D_j = \frac{\sqrt{\max(\hat F_j,\,\varepsilon)}}{\frac{1}{p}\sum_{j'}\sqrt{\max(\hat F_{j'},\,\varepsilon)}},
$$

with $\varepsilon = 10^{-8}$ as a floor.

### Interpretation

$\hat F_j$ measures the information content of the data about parameter $\theta_j$: how much the log-likelihood changes per unit change in $\theta_j$.  The square-root scaling $D_j \propto \sqrt{F_j}$ converts curvature (second-order information) into a length scale.  Parameters with high Fisher information are compressed in kernel space, making the kernel more sensitive to movement along directions that the data constrains tightly.

This is the natural Riemannian metric on the statistical manifold.  In the Fisher geometry, the kernel becomes locally isotropic with respect to the information content of each parameter.

### The instability problem

The score involves division by $p(x_n|\theta_\text{ref})$.  When some observations have very low probability under the reference parameters---common in multi-stage phase-type distributions where the tail is thin---the scores blow up:

$$
s_{nj} = \frac{\partial_j p(x_n|\theta)}{p(x_n|\theta)} \to \infty \quad \text{as} \quad p(x_n|\theta) \to 0.
$$

A single outlier observation can dominate the entire Fisher diagonal, producing a scaling vector that is driven by numerical noise rather than genuine parameter sensitivity.  The $\varepsilon$ floor prevents division by zero but does not prevent large-but-finite blowup.

### Cost

The Fisher method requires the same $2p$ finite-difference evaluations as the Jacobian method, but additionally requires evaluating and storing the per-observation PMF vector and performing $N \times p$ divisions.  For large observation sets this adds nontrivial overhead.

## Comparison

| Property | Moment Jacobian | Fisher |
|:---|:---|:---|
| What it measures | Moment sensitivity to each parameter | Information content per parameter |
| Numerical stability | Division-free | Divides by PMF; blows up in tails |
| Model evaluations | $2p + 1$ | $2p + 1$ |
| Per-observation work | None | $O(Np)$ divisions |
| Theoretical basis | Sensitivity analysis | Information geometry |
| When to prefer | Default; multi-scale rates | When you specifically want the Fisher metric and the PMF is well-behaved at the reference |

## Usage

```python
from phasic import Graph, SVGD

graph = Graph(my_callback, ...)
model = graph.pmf_and_moments_from_graph(nr_moments=2)

# Default: Moment Jacobian (recommended)
result = graph.svgd(observed_data, theta_dim=2, preconditioner='auto')

# Explicit Moment Jacobian
result = graph.svgd(observed_data, theta_dim=2, preconditioner='jacobian')

# Fisher preconditioning
result = graph.svgd(observed_data, theta_dim=2, preconditioner='fisher')

# No preconditioning
result = graph.svgd(observed_data, theta_dim=2, preconditioner=None)
```

For fine-grained control, instantiate a preconditioner directly:

```python
from phasic import MomentJacobianPreconditioner, FisherPreconditioner

precond = MomentJacobianPreconditioner(
    model=model,
    observed_data=data,
    theta_dim=2,
    param_transform=jax.nn.softplus,
    epsilon=1e-6   # custom floor
)

result = graph.svgd(observed_data, theta_dim=2, preconditioner=precond)
```
