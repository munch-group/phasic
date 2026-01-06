# Optimizers for SVGD

Phasic provides several optimizers for SVGD that offer adaptive per-parameter learning rates. These can be particularly useful when gradients have vastly different scales across parameters or when fixed step sizes cause oscillation.

**Available optimizers:**
- **Adam** - Adaptive moment estimation (recommended default)
- **SGDMomentum** - SGD with momentum
- **RMSprop** - Root mean square propagation
- **Adagrad** - Adaptive gradient algorithm

## How Adam Works

Adam (Adaptive Moment Estimation) maintains running estimates of two quantities for each parameter:

1. **First moment (m)**: An exponentially weighted average of past gradients (momentum)
2. **Second moment (v)**: An exponentially weighted average of past squared gradients (gradient variance)

At each iteration, Adam computes:

```
m ← β₁ · m + (1 - β₁) · gradient
v ← β₂ · v + (1 - β₂) · gradient²

m̂ = m / (1 - β₁ᵗ)     # Bias correction
v̂ = v / (1 - β₂ᵗ)     # Bias correction

update = lr · m̂ / (√v̂ + ε)
```

The key insight is that each parameter gets its own effective learning rate:
- Parameters with consistently large gradients → larger v̂ → smaller effective step
- Parameters with consistently small gradients → smaller v̂ → larger effective step

This automatic scaling helps when different parameters naturally have different gradient magnitudes.

## Adam vs Fixed Learning Rate

| Fixed Learning Rate | Adam |
|---------------------|------|
| Same step size for all parameters | Adaptive step size per parameter |
| Can oscillate when gradients vary widely | Dampens oscillations via momentum |
| Requires careful tuning | More robust to initial choice |
| Simpler, less overhead | Tracks additional state (m, v) |

**When fixed learning rate works well:**
- Well-behaved optimization landscapes
- Parameters with similar gradient scales
- When you've tuned the learning rate carefully

**When Adam helps:**
- Large datasets causing large gradient magnitudes
- Parameters with vastly different scales
- "Shark teeth" oscillation patterns in convergence
- When you want reasonable results without extensive tuning

## Parameters

```python
from phasic import Adam

optimizer = Adam(
    learning_rate=0.001,  # Base learning rate (α)
    beta1=0.9,            # Momentum decay (typical: 0.9)
    beta2=0.999,          # Gradient variance decay (typical: 0.999)
    epsilon=1e-8          # Numerical stability constant
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.001 | Base learning rate, scaled per-parameter by Adam |
| `beta1` | 0.9 | Decay rate for first moment (momentum). Higher = more smoothing |
| `beta2` | 0.999 | Decay rate for second moment. Higher = longer memory of gradient magnitudes |
| `epsilon` | 1e-8 | Small constant to prevent division by zero |

### Tuning Guidelines

- **learning_rate**: Start with 0.001-0.01. If convergence is too slow, increase. If unstable, decrease.
- **beta1**: 0.9 works well for most cases. Lower values (0.8) give less momentum.
- **beta2**: 0.999 is standard. Lower values (0.99) adapt faster to gradient changes.
- **epsilon**: Rarely needs adjustment. Increase to 1e-7 if you see numerical issues.

## Usage Examples

### Basic Usage

```python
from phasic import SVGD, Adam

# Create optimizer
optimizer = Adam(learning_rate=0.01)

# Use with SVGD
svgd = SVGD(
    model=model,
    observed_data=observations,
    theta_dim=2,
    optimizer=optimizer,
    n_particles=50,
    n_iterations=200
)
svgd.fit()
```

### Via Graph.svgd()

```python
from phasic import Adam

optimizer = Adam(learning_rate=0.01)
svgd = joint_graph.svgd(
    obs_indices,
    theta_dim=2,
    optimizer=optimizer,
    n_particles=50,
    n_iterations=200
)
```

### With Fixed Parameters

```python
from phasic import Adam

optimizer = Adam(learning_rate=0.01)
svgd = joint_graph.svgd(
    obs_indices,
    theta_dim=2,
    fixed=[(1, 0.01)],  # Fix theta[1] at 0.01
    optimizer=optimizer,
    n_particles=50,
    n_iterations=200
)
```

When using fixed parameters, Adam operates only on the learnable dimensions. The optimizer state shape matches the number of learnable parameters.

## When to Use Adam

**Consider Adam when you observe:**

1. **Oscillating convergence** ("shark teeth" pattern in loss)
2. **Different parameters converging at different rates**
3. **Sensitivity to learning rate choice**
4. **Large datasets** causing large gradient magnitudes

**Stick with fixed learning rate when:**

1. **Convergence is already smooth and fast**
2. **You've tuned the learning rate well for your problem**
3. **You want minimal computational overhead**
4. **The optimization landscape is well-behaved**

## Notes

- When using Adam, the `learning_rate` parameter passed to SVGD is ignored in favor of the optimizer's learning rate.
- Step size schedules (`ExpStepSize`, `AdaptiveStepSize`) are not used when Adam is enabled.
- Adam is fully compatible with JAX's JIT compilation.

## Other Optimizers

### SGDMomentum

SGD with momentum accumulates velocity in directions of persistent gradient descent, helping accelerate convergence and dampen oscillations.

```python
from phasic import SGDMomentum

optimizer = SGDMomentum(
    learning_rate=0.01,  # Step size
    momentum=0.9         # Momentum coefficient (0.9 standard, 0.99 high)
)
```

**Update rule:** `v = momentum * v + gradient; params += lr * v`

### RMSprop

RMSprop divides the learning rate by an exponentially decaying average of squared gradients, adapting per-parameter.

```python
from phasic import RMSprop

optimizer = RMSprop(
    learning_rate=0.001,  # Base learning rate
    decay=0.99,           # Decay rate for squared gradient average
    epsilon=1e-8          # Numerical stability
)
```

**Update rule:** `v = decay * v + (1 - decay) * gradient²; params += lr * gradient / (√v + ε)`

### Adagrad

Adagrad accumulates the sum of squared gradients, giving smaller learning rates to parameters with large accumulated gradients.

```python
from phasic import Adagrad

optimizer = Adagrad(
    learning_rate=0.01,  # Base learning rate
    epsilon=1e-8         # Numerical stability
)
```

**Update rule:** `G += gradient²; params += lr * gradient / (√G + ε)`

**Note:** Adagrad's learning rate decays over time as G accumulates. For long runs, RMSprop or Adam may perform better.

## Optimizer Comparison

| Optimizer | Adaptive LR | Momentum | Best For |
|-----------|-------------|----------|----------|
| Fixed LR | No | No | Simple, well-tuned problems |
| SGDMomentum | No | Yes | Accelerating convergence |
| RMSprop | Yes | No | Non-stationary objectives |
| Adagrad | Yes | No | Sparse gradients |
| Adam | Yes | Yes | General purpose (recommended) |

## References

- [Kingma and Ba (2014)](https://arxiv.org/abs/1412.6980) - Adam: A Method for Stochastic Optimization. arXiv:1412.6980.
- [Hinton (2012)](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) - RMSprop. Coursera: Neural Networks for Machine Learning.
- [Duchi et al. (2011)](https://jmlr.org/papers/v12/duchi11a.html) - Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. JMLR 12:2121-2159.
