# Combining PDF and Moment-Based Inference for Non-Smooth Population Histories

## The Problem

In epoch-wise time-inhomogeneous coalescent models:

- **Full distribution** (e.g., TMRCA): Can compute exact PDF via `distribution_context().step()`
- **Feature marginals** (singletons, doubletons, etc.): Can only compute **exact moments** via `accumulated_occupancy()` and reward transformation

The goal is to combine these information sources for inference on population size history $N(t)$, which may be **non-smooth** (jumps, discontinuities, arbitrary shape).

### The Computational Setup (from phasic)

```python
# Full distribution: PDF available via stepwise computation
ctx = graph.distribution_context()
while ctx.cdf() < 0.999:
    pdf_value = ctx.pdf()
    time = ctx.time()
    # Can change parameters between steps for time-inhomogeneity
    graph.update_weights([1.0 / N_current])
    ctx.step()

# Feature marginals: Only moments available
rewards = graph.states()[:, :-1]  # Reward vectors for each feature
for i in range(n_features):
    transformed = graph.reward_transform(rewards[:, i])
    feature_moments[i] = transformed.moments(nr_moments)
```

---

## Data Structure

- **Individual observations** for both full distribution and all features
- **Exact moments** computable to arbitrary order via phase-type machinery
- **Features assumed independent** given parameters (conditional on the genealogy)
- **Large sample size** (n ≥ 10,000 observations)

---

## The Naive Approach: Why It's Problematic

### Initial Proposal

Combine PDF likelihood with moment-matching penalty:

$$\mathcal{L}(\theta) = \underbrace{\sum_{i} \log f(t_i; \theta)}_{\text{PDF likelihood}} - \lambda \underbrace{\|\hat{m} - \mu(\theta)\|^2}_{\text{moment penalty}}$$

### The Efficient Weighting

For large samples, the asymptotically efficient combination uses the **moment covariance matrix**:

$$\mathcal{L}(\theta) = \sum_{i} \log f(t_i; \theta) - \frac{1}{2}(\hat{m} - \mu(\theta))^T \Sigma^{-1}(\theta) (\hat{m} - \mu(\theta))$$

where:
$$[\Sigma(\theta)]_{kl} = \frac{1}{n}\left(\mu_{k+l}(\theta) - \mu_k(\theta)\mu_l(\theta)\right)$$

This requires moments up to order $2K$ to weight the first $K$ moments properly.

### The Critical Assumption That Fails

This efficiency result assumes:

1. **Regularity conditions** on the likelihood (smoothness, differentiability)
2. **Local asymptotic normality** of the model
3. **Correct model specification**

**But the true population history $N(t)$ can be any non-smooth function!**

---

## The Fundamental Problem: Model Misspecification

### Many Functions → Same Moments

The key insight you identified:

> "Many non-smooth distributions may have the same smooth moment generating function."

This means moments alone **cannot distinguish** between:
- Smooth population size decline
- Step-wise population size changes
- Oscillating population sizes
- Bottlenecks followed by expansion

All could produce **identical moments**.

### What Happens Under Misspecification

When the true data-generating process $P_0$ is not in the model family $\{P_\theta\}$:

- MLE converges to the **pseudo-true parameter**: $\theta^* = \arg\min_\theta KL(P_0 \| P_\theta)$
- This minimizes Kullback-Leibler divergence to the "closest" model
- **Not necessarily unbiased** for any meaningful quantity
- The estimator finds the best approximation within the model class, not the truth

### The Identifiability Gap

| What We Want | What We Can Get |
|--------------|-----------------|
| True $N(t)$ (possibly non-smooth) | Smooth $\tilde{N}(t)$ matching moments |
| Full distributional information | Polynomial projections (moments) |
| Unique solution | Equivalence class of solutions |

---

## What Information Does Each Source Provide?

### Full Distribution PDF (e.g., TMRCA)

The PDF contains **all information** about the marginal distribution:

- **Shape**: Modes, tails, skewness, kurtosis
- **Fine structure**: Bumps, kinks from population size changes
- **Local features**: If $N(t)$ jumps at time $t^*$, the density often shows a **kink or slope change** at $t^*$

**Key property**: The PDF **can detect non-smoothness** in the underlying process.

### Feature Moments (singletons, doubletons)

Moments are **projections** onto polynomial basis functions:

- $E[X]$: Location (mean coalescence time)
- $E[X^2]$: Spread (variance + mean²)
- $E[X^k]$: Tail behavior for large $k$

**Key limitation**: Moments are **insensitive to local structure** — they integrate over the entire distribution, averaging out discontinuities.

### The Complementarity

| Aspect | PDF | Moments |
|--------|-----|---------|
| Local structure | ✓ Captures | ✗ Averages out |
| Global properties | ✓ Implicit | ✓ Explicit |
| Non-smoothness detection | ✓ Yes | ✗ No |
| Computational availability | Limited (full dist. only) | All features |

---

## Theoretical Frameworks for Combination

### Option 1: Moment Constraints as Hard Constraints

Treat moments as exact constraints:

$$\hat{\theta} = \arg\max_\theta p(\text{TMRCA obs} | \theta) \quad \text{s.t.} \quad \mu_k(\theta) = \hat{m}_k$$

**Problem**: May be over-constrained or infeasible due to sampling noise in $\hat{m}_k$.

### Option 2: Soft Constraints (Current SVGD Approach)

Penalize moment discrepancy:

$$\log p(\theta | \text{data}) \propto \log p(\text{TMRCA} | \theta) + \log p(\theta) - \lambda \|\mu(\theta) - \hat{m}\|^2_{\Sigma^{-1}}$$

**Issue**: Choice of $\lambda$ is ad hoc. What's the "right" tradeoff?

### Option 3: Empirical Likelihood (Most Principled)

Empirical likelihood makes **no parametric assumptions** about feature distributions:

$$\mathcal{L}_{EL}(\theta) = \max_{p_1,...,p_n} \sum_i \log p_i \quad \text{s.t.} \quad \sum_i p_i = 1, \quad \sum_i p_i (x_i^k - \mu_k(\theta)) = 0$$

This finds the **nonparametric distribution** closest to empirical that satisfies moment constraints.

**Advantages**:
- No smoothness assumptions on true distribution
- Automatic "optimal" weighting
- Wilks-type confidence regions

**Disadvantage**: Computationally more complex.

### Option 4: Approximate Bayesian Computation (ABC) Spirit

$$p(\theta | \text{data}) \propto p(\text{TMRCA} | \theta) \cdot K_\epsilon(d(\mu(\theta), \hat{m})) \cdot p(\theta)$$

Accept parameters whose moments are within $\epsilon$ of observed. The kernel $K_\epsilon$ softens this to a smooth penalty.

---

## The Recommended Approach

### Hierarchical Objective

```
log_posterior(θ) = log_likelihood_pdf(θ, tmrca_obs)     # Exact: trust this fully
                 + log_prior(θ)                          # Regularization
                 - λ · moment_discrepancy(θ, features)   # Soft constraint
```

### Proper Moment Weighting

Use the covariance-weighted discrepancy:

$$\text{moment\_discrepancy}(\theta) = (\hat{m} - \mu(\theta))^T \Sigma^{-1}(\theta) (\hat{m} - \mu(\theta))$$

where the covariance matrix is computed from higher moments:

$$\Sigma_{kl} = \frac{1}{n}\left(\mu_{k+l} - \mu_k \mu_l\right)$$

### Choosing the Regularization Weight λ

The weight controls the tradeoff:

| λ Value | Behavior |
|---------|----------|
| λ → 0 | Pure MLE on TMRCA (ignores features) |
| λ → ∞ | Exact moment matching (may sacrifice PDF fit) |
| λ optimal | Balances both information sources |

**Principled approaches to choose λ**:

1. **Cross-validation**: Hold out TMRCA observations, choose λ minimizing prediction error
2. **Simulation calibration**: Generate data from known non-smooth histories, tune λ for best recovery
3. **Empirical Bayes**: Treat λ as hyperparameter, estimate from data

---

## Honest Interpretation of Results

Given the fundamental identifiability issue, results should be interpreted as:

> "We estimate the **smooth population history** $\tilde{N}(t)$ that:
> 1. Maximizes the TMRCA likelihood
> 2. Approximately matches observed feature moments
>
> This represents a **projection** of the true (possibly non-smooth) history onto the space of smooth histories representable by our model."

### What We Can Claim

- ✓ The estimated $\tilde{N}(t)$ is the best smooth approximation given our model class
- ✓ Feature moments are approximately matched
- ✓ TMRCA distribution is well-fitted

### What We Cannot Claim

- ✗ We have recovered the true $N(t)$ if it's non-smooth
- ✗ The solution is unique (moment-equivalent alternatives may exist)
- ✗ Uncertainty quantification captures model misspecification

---

## The Saving Grace: PDF Detects Non-Smoothness

The TMRCA PDF is crucial because it **can** detect non-smooth structure:

- Sudden population size changes create **kinks** in the density
- Bottlenecks create **modes** in the coalescence time distribution
- The PDF likelihood will **resist** overly-smooth solutions

So the combination leverages:
- **Moments**: Ensure global properties are correct (mean times, variances)
- **PDF**: Capture local structure that moments miss

The PDF acts as a **regularizer against over-smoothing** while moments ensure **global calibration**.

---

## Implementation Considerations

### Computing Moment Covariance

```python
def compute_moment_covariance(model_moments, higher_moments, n_samples):
    """
    Compute covariance matrix of sample moments.

    Cov(m_k, m_l) = (1/n) * (E[X^{k+l}] - E[X^k] * E[X^l])

    Parameters
    ----------
    model_moments : array, shape (K,)
        First K raw moments [E[X], E[X²], ..., E[X^K]]
    higher_moments : array, shape (2K,)
        Moments up to order 2K for covariance computation
    n_samples : int
        Number of observations

    Returns
    -------
    Sigma : array, shape (K, K)
        Covariance matrix of sample moments
    """
    K = len(model_moments)
    Sigma = np.zeros((K, K))

    for k in range(K):
        for l in range(K):
            # E[X^{k+1} · X^{l+1}] = E[X^{k+l+2}]
            mu_kl = higher_moments[k + l + 1]  # E[X^{(k+1)+(l+1)}]
            mu_k = model_moments[k]             # E[X^{k+1}]
            mu_l = model_moments[l]             # E[X^{l+1}]

            Sigma[k, l] = (mu_kl - mu_k * mu_l) / n_samples

    return Sigma
```

### Combined Objective Function

```python
def combined_log_posterior(theta, tmrca_obs, feature_obs_dict, nr_moments, lambda_weight):
    """
    Combined objective for PDF + moment-based inference.
    """
    # 1. TMRCA PDF likelihood (exact)
    log_lik = jnp.sum(jnp.log(tmrca_pdf(tmrca_obs, theta) + 1e-300))

    # 2. Prior
    log_prior = compute_log_prior(theta)

    # 3. Feature moment penalties (properly weighted)
    moment_penalty = 0.0

    for feature_name, obs in feature_obs_dict.items():
        n_j = len(obs)

        # Sample moments
        sample_mom = jnp.array([jnp.mean(obs**k) for k in range(1, nr_moments + 1)])

        # Model moments (exact)
        model_mom = compute_feature_moments(feature_name, theta, nr_moments)

        # Covariance (requires 2*nr_moments)
        higher_mom = compute_feature_moments(feature_name, theta, 2 * nr_moments)
        Sigma = compute_moment_covariance(model_mom, higher_mom, n_j)

        # Mahalanobis distance
        diff = sample_mom - model_mom
        moment_penalty += diff @ jnp.linalg.solve(Sigma, diff)

    return log_lik + log_prior - lambda_weight * moment_penalty
```

---

## Theoretical Foundations for Sound Inference

Given your goals of **accuracy**, **confidence estimation**, and **sound theoretical foundation**, we need to carefully consider what can be rigorously justified.

### The Core Question

When the true $N(t)$ is non-smooth but we fit a smooth phase-type model, what can we claim about:
1. **Point estimates**: What parameter values do we converge to?
2. **Uncertainty quantification**: Are confidence intervals valid?
3. **Prediction**: How well do we predict new observations?

### Key Theoretical Results

#### 1. Quasi-Maximum Likelihood Theory ([White, 1982](https://doi.org/10.2307/1912526))

Even under misspecification, MLE converges to a well-defined limit:

$$\hat{\theta}_n \xrightarrow{p} \theta^* = \arg\min_\theta KL(P_0 \| P_\theta)$$

where $P_0$ is the true distribution and $P_\theta$ is our model.

**Interpretation**: We find the model that is "closest" to truth in Kullback-Leibler divergence.

**For the PDF component**: This is exactly what MLE on TMRCA observations achieves.

#### 2. Generalized Method of Moments Theory ([Hansen, 1982](https://doi.org/10.2307/1912775))

For moment conditions $g(\theta) = E[m(X)] - \mu(\theta) = 0$:

The GMM estimator $\hat{\theta}_{GMM}$ is:
- **Consistent** for $\theta^*$ where $\mu(\theta^*) = E_{P_0}[m(X)]$ (the true moments)
- **Asymptotically normal**: $\sqrt{n}(\hat{\theta} - \theta^*) \xrightarrow{d} N(0, V)$
- **Valid confidence intervals** based on the sandwich estimator

**Critical point**: GMM is valid even when the model is misspecified, as long as the moment conditions identify a unique $\theta^*$.

#### 3. The Sandwich Covariance Estimator

Under misspecification, the naive covariance estimate is wrong. The correct "sandwich" form is:

$$V = (D^T W D)^{-1} D^T W \Omega W D (D^T W D)^{-1}$$

where:
- $D = \partial \mu(\theta) / \partial \theta$ (Jacobian of model moments)
- $W$ = weighting matrix (optimally $\Omega^{-1}$)
- $\Omega$ = true covariance of moment conditions under $P_0$

**Key insight**: We can estimate $\Omega$ from the data without knowing $P_0$!

### A Rigorous Combined Estimator

Given the theory, here is a **defensible** approach:

#### Step 1: Define the Estimand

We estimate $\theta^*$ defined as the solution to:

$$\theta^* = \arg\min_\theta \left[ -E_{P_0}[\log f_\theta(T)] + \frac{1}{2} \|\mu(\theta) - \mu_0\|^2_{W} \right]$$

where:
- $T$ = TMRCA observations
- $\mu_0$ = true feature moments (estimated by sample moments)
- $W$ = weighting matrix
- $f_\theta$ = phase-type PDF under parameter $\theta$

This is the parameter that **best fits the TMRCA PDF while approximately matching feature moments**.

#### Step 2: Estimation

Sample analog:

$$\hat{\theta} = \arg\max_\theta \left[ \sum_i \log f_\theta(t_i) - \frac{n}{2} (\hat{m} - \mu(\theta))^T W (\hat{m} - \mu(\theta)) \right]$$

#### Step 3: Covariance Estimation (Sandwich Form)

The asymptotic covariance of $\hat{\theta}$ is:

$$\text{Var}(\hat{\theta}) = \frac{1}{n} H^{-1} V H^{-1}$$

where:
- $H$ = Hessian of the objective at $\hat{\theta}$
- $V$ = variance of the score (gradient) under $P_0$

Both can be estimated from data:

```python
def sandwich_covariance(theta_hat, pdf_obs, feature_obs, model):
    """
    Compute sandwich covariance estimator.

    Valid even under model misspecification.
    """
    n = len(pdf_obs)

    # Compute individual score contributions
    scores = []
    for i in range(n):
        score_i = compute_score(theta_hat, pdf_obs[i], feature_obs[i], model)
        scores.append(score_i)
    scores = np.array(scores)

    # V = empirical covariance of scores (the "meat")
    V = np.cov(scores.T) * n

    # H = Hessian of objective (the "bread")
    H = compute_hessian(theta_hat, pdf_obs, feature_obs, model)

    # Sandwich: H^{-1} V H^{-1}
    H_inv = np.linalg.inv(H)
    sandwich_cov = H_inv @ V @ H_inv / n

    return sandwich_cov
```

### Optimal Weighting: Two-Step GMM

For efficiency, use the **two-step GMM** procedure:

1. **First step**: Estimate with identity weighting $W = I$
   $$\hat{\theta}^{(1)} = \arg\max_\theta \left[ \ell_{PDF}(\theta) - \frac{n}{2} \|\hat{m} - \mu(\theta)\|^2 \right]$$

2. **Estimate optimal weights**:
   $$\hat{W} = \hat{\Sigma}^{-1}$$
   where $\hat{\Sigma}$ is estimated from residuals at $\hat{\theta}^{(1)}$

3. **Second step**: Re-estimate with optimal weights
   $$\hat{\theta}^{(2)} = \arg\max_\theta \left[ \ell_{PDF}(\theta) - \frac{n}{2} (\hat{m} - \mu(\theta))^T \hat{W} (\hat{m} - \mu(\theta)) \right]$$

**Theorem** ([Hansen, 1982](https://doi.org/10.2307/1912775)): The two-step GMM estimator achieves the semiparametric efficiency bound among all estimators using these moment conditions.

### What We Can Rigorously Claim

| Claim | Validity | Conditions |
|-------|----------|------------|
| $\hat{\theta} \to \theta^*$ (consistency) | ✓ Valid | Standard regularity |
| Confidence intervals cover $\theta^*$ | ✓ Valid | Sandwich covariance |
| $\theta^* = \theta_{\text{true}}$ | ✗ Not guaranteed | Requires correct specification |
| Predictions for new TMRCA | ✓ Valid | By construction |
| Predictions for feature moments | ✓ Valid | By moment matching |

### The Interpretation

**What $\theta^*$ represents**:

> $\theta^*$ is the parameter of the smooth phase-type model that:
> 1. Minimizes KL divergence to the true TMRCA distribution
> 2. Exactly matches the true feature moments
>
> This is the **best smooth approximation** to the true (possibly non-smooth) population history, where "best" is defined by the combined PDF + moment criterion.

**What confidence intervals mean**:

> A 95% CI for $\theta^*$ covers the true $\theta^*$ in 95% of repeated samples.
> It does **not** necessarily cover the "true" parameter if the model is wrong.
> But $\theta^*$ is a well-defined, interpretable quantity.

---

## Practical Implementation

### The Combined Objective (Properly Scaled)

```python
def combined_objective(theta, tmrca_obs, feature_obs_dict, nr_moments, W=None):
    """
    Combined PDF + moment objective with proper scaling.

    Parameters
    ----------
    theta : array
        Model parameters
    tmrca_obs : array, shape (n_tmrca,)
        TMRCA observations
    feature_obs_dict : dict
        {feature_name: observations} for each feature
    nr_moments : int
        Number of moments per feature
    W : array, optional
        Weighting matrix for moments (default: identity)

    Returns
    -------
    objective : float
        Combined log-likelihood + moment penalty
    """
    n_tmrca = len(tmrca_obs)

    # 1. PDF log-likelihood (normalized by sample size)
    log_lik = jnp.sum(jnp.log(tmrca_pdf(tmrca_obs, theta) + 1e-300)) / n_tmrca

    # 2. Moment penalty
    all_sample_moments = []
    all_model_moments = []

    for feature_name, obs in feature_obs_dict.items():
        # Sample moments
        sample_mom = jnp.array([jnp.mean(obs**k) for k in range(1, nr_moments + 1)])
        all_sample_moments.append(sample_mom)

        # Model moments (exact from phase-type)
        model_mom = compute_feature_moments(feature_name, theta, nr_moments)
        all_model_moments.append(model_mom)

    sample_moments = jnp.concatenate(all_sample_moments)
    model_moments = jnp.concatenate(all_model_moments)

    # Moment discrepancy
    diff = sample_moments - model_moments

    if W is None:
        W = jnp.eye(len(diff))

    moment_penalty = 0.5 * diff @ W @ diff

    return log_lik - moment_penalty
```

### Computing the Sandwich Covariance

```python
def compute_sandwich_covariance(theta_hat, tmrca_obs, feature_obs_dict,
                                 nr_moments, W):
    """
    Sandwich covariance estimator for valid inference under misspecification.
    """
    n_total = len(tmrca_obs)
    n_params = len(theta_hat)

    # Compute score for each observation
    def score_i(theta, tmrca_i, feature_dict_i):
        return jax.grad(lambda t: combined_objective(t, [tmrca_i], feature_dict_i, nr_moments, W))(theta)

    scores = []
    for i in range(n_total):
        # Extract i-th observation from each feature
        feature_i = {k: v[[i]] for k, v in feature_obs_dict.items()}
        score = score_i(theta_hat, tmrca_obs[i], feature_i)
        scores.append(score)

    scores = jnp.array(scores)

    # "Meat": empirical covariance of scores
    V = jnp.cov(scores.T) * n_total

    # "Bread": Hessian of objective
    H = jax.hessian(lambda t: combined_objective(t, tmrca_obs, feature_obs_dict, nr_moments, W))(theta_hat)
    H = -H * n_total  # Convert to information matrix convention

    # Sandwich
    H_inv = jnp.linalg.inv(H)
    sandwich = H_inv @ V @ H_inv / n_total

    return sandwich
```

### Two-Step Efficient Estimation

```python
def two_step_gmm(tmrca_obs, feature_obs_dict, nr_moments, initial_theta):
    """
    Two-step GMM for efficient estimation.
    """
    # Step 1: Identity weighting
    W1 = jnp.eye(nr_moments * len(feature_obs_dict))

    theta_step1 = optimize(
        lambda t: -combined_objective(t, tmrca_obs, feature_obs_dict, nr_moments, W1),
        initial_theta
    )

    # Estimate optimal weighting from residuals
    sample_moments = compute_all_sample_moments(feature_obs_dict, nr_moments)
    model_moments = compute_all_model_moments(theta_step1, feature_obs_dict, nr_moments)

    # Estimate moment covariance (can use bootstrap or analytical)
    Sigma_hat = estimate_moment_covariance(feature_obs_dict, nr_moments)
    W2 = jnp.linalg.inv(Sigma_hat)

    # Step 2: Optimal weighting
    theta_step2 = optimize(
        lambda t: -combined_objective(t, tmrca_obs, feature_obs_dict, nr_moments, W2),
        theta_step1
    )

    # Compute sandwich covariance
    cov = compute_sandwich_covariance(theta_step2, tmrca_obs, feature_obs_dict, nr_moments, W2)

    return theta_step2, cov
```

---

## Summary: A Sound Theoretical Framework

### What We Estimate

The parameter $\theta^*$ that defines the smooth phase-type distribution which:
1. **Best approximates** the true TMRCA distribution (in KL divergence)
2. **Matches** the true feature moments exactly

### Why This Is Defensible

1. **GMM theory** guarantees consistency and asymptotic normality for $\theta^*$
2. **Sandwich covariance** provides valid confidence intervals even under misspecification
3. **Two-step procedure** achieves optimal efficiency among moment-based estimators
4. **PDF likelihood** captures fine structure (modes, kinks) that moments miss

### What We Cannot Claim

- The estimated smooth $N(t)$ equals the true non-smooth $N(t)$
- Confidence intervals cover the "true" parameter (only $\theta^*$)
- The model is correctly specified

### Practical Recommendations

1. **Use two-step GMM** for efficiency
2. **Report sandwich standard errors** for valid inference
3. **Validate via simulation** from known non-smooth histories
4. **Interpret carefully**: $\theta^*$ is the best smooth approximation

---

## Parameterizing Population Size History: 25 Parameters

In realistic demographic inference, we need sufficient flexibility to capture complex population histories. A typical scenario requires **25 independent epochs** to span the time range of interest. This raises the question: how should we parameterize $N(t)$ with limited degrees of freedom?

### The Challenge

Population size history $N(t)$ must be:
1. **Piecewise time-homogeneous** (epoch-wise constant for computational tractability)
2. **Flexible enough** to capture bottlenecks, expansions, and varying rates of change
3. **Constrained enough** to enable robust inference with finite data

### Three Approaches to 25-Parameter Models

#### Approach 1: Independent Epoch Values

The simplest parameterization: one free parameter per epoch.

$$N_i = \theta_i \quad \text{for } i = 1, \ldots, 25$$

**Properties:**
- **Flexibility**: Maximum — any piecewise-constant history is representable
- **Smoothness**: None imposed — adjacent epochs are independent
- **Identifiability**: May be problematic with limited data
- **Parameters**: 25 free parameters

**Mathematical structure:**
- No relationship between $N_i$ and $N_{i+1}$
- Can represent arbitrary jumps between epochs
- Variance in estimates grows with epoch index (deeper time = less data)

#### Approach 2: Phase-Type Auxiliary Chain

Use a continuous-time Markov chain to model transitions between population size "regimes."

**The Idea**: Instead of 25 independent values, define an auxiliary process $Z(t)$ on a small state space $\{1, 2, \ldots, K\}$ with $K \ll 25$. The population size depends on the auxiliary state:

$$N(t) = N_{Z(t)}$$

where $Z(t)$ is governed by a sub-intensity matrix $Q$ of dimension $K \times K$.

**Parameters** (for $K$ states):
- $K$ population size values: $N_1, \ldots, N_K$
- $K(K-1)$ off-diagonal transition rates in $Q$
- Total: $K + K(K-1) = K^2$ parameters

For $K = 5$: 25 parameters (matching independent epochs)

**Mathematical properties:**
- **Smoothness**: Markov property induces autocorrelation in $N(t)$
- **Transitions**: Population changes occur at exponentially-distributed times
- **Equilibrium**: Can reach stationary distribution (if desired)
- **Identifiability**: Fewer effective degrees of freedom due to structure

**Advantages over independent epochs:**
1. **Implicit regularization**: Adjacent time points are correlated through Markov dynamics
2. **Biological interpretability**: Regime-switching models have ecological meaning
3. **Extrapolation**: Better behavior outside observed time range
4. **Parsimony**: Can capture complex patterns with fewer effective parameters

**Specific auxiliary chain options:**

**(a) Two-State Chain (4 parameters)**
$$Q = \begin{pmatrix} -\alpha & \alpha \\ \beta & -\beta \end{pmatrix}$$

- States: "High $N$" and "Low $N$"
- Parameters: $N_{\text{high}}, N_{\text{low}}, \alpha, \beta$
- Suitable for: Boom-bust dynamics, expansion-contraction

**(b) Birth-Death Chain on $K$ States (3$K$-2 parameters)**
$$Q_{i,i+1} = \lambda_i, \quad Q_{i,i-1} = \mu_i$$

- Only nearest-neighbor transitions
- Parameters: $\{N_i\}_{i=1}^K, \{\lambda_i\}_{i=1}^{K-1}, \{\mu_i\}_{i=2}^K$
- Suitable for: Gradual size changes with occasional reversals

**(c) Coxian Distribution (~2$K$ parameters)**
$$Q = \begin{pmatrix}
-(\lambda_1 + \mu_1) & \lambda_1 & 0 & \cdots \\
0 & -(\lambda_2 + \mu_2) & \lambda_2 & \cdots \\
\vdots & & \ddots & \\
0 & \cdots & 0 & -\mu_K
\end{pmatrix}$$

- Sequential progression through states (no backwards transitions)
- Parameters: $\{N_i\}, \{\lambda_i\}, \{\mu_i\}$
- Suitable for: Monotonic trends with varying rates

**(d) OU-like Discretization (~$K$ + 3 parameters)**

Discretize an Ornstein-Uhlenbeck process:
$$dX_t = \theta(\mu - X_t)dt + \sigma dW_t$$

- Mean-reverting to equilibrium $\mu$
- Parameters: $\theta$ (reversion rate), $\mu$ (mean), $\sigma$ (volatility), plus $K$ discretization states
- Suitable for: Fluctuations around carrying capacity

#### Approach 3: Spline Parameterization

Model $\log N(t)$ as a spline function.

**B-Spline with $K$ knots:**
$$\log N(t) = \sum_{j=1}^{K+d-1} c_j B_j^d(t)$$

where $B_j^d$ are B-spline basis functions of degree $d$.

**Properties:**
- **Smoothness**: $C^{d-1}$ continuous (cubic splines: $C^2$ smooth)
- **Flexibility**: Controlled by number of knots
- **Parameters**: $K + d - 1$ coefficients
- **Computation**: Not directly compatible with phase-type formulation

For 25 parameters with cubic splines: ~22 knots

**Comparison Table:**

| Property | Independent | Phase-Type Aux | Splines |
|----------|-------------|----------------|---------|
| Discontinuities | ✓ Natural | ✓ Natural | ✗ Smooth only |
| Parameter count | 25 | ~25 (K²) | ~25 |
| Smoothness | None | Markov-induced | Explicit |
| Phase-type compatible | ✓ Yes | ✓ Yes | ✗ Requires discretization |
| Biological interpretation | Direct | Regime switching | Mathematical |
| Deep time extrapolation | Poor | Good | Depends on boundary |
| Identifiability | May struggle | Better constrained | Good |
| Captures bottlenecks | ✓ Yes | ✓ Yes | ✓ Yes (if knot placed) |
| Captures rapid changes | ✓ Yes | ✓ Yes | ✗ Smoothed out |

### Why Phase-Type Auxiliary Chain is Attractive

1. **Universal approximation**: Phase-type distributions can approximate any distribution on $\mathbb{R}_+$ arbitrarily well. This extends to $N(t)$ histories.

2. **Computational compatibility**: The auxiliary chain naturally integrates with phase-type computation — the combined state space is $(n, z)$ where $n$ is lineage count and $z$ is auxiliary state.

3. **Flexibility vs. structure tradeoff**: With $K=5$ states:
   - 25 parameters total
   - But only ~10 "effective" degrees of freedom due to Markov structure
   - Remaining parameters control transition dynamics

4. **Captures non-smooth features**:
   - Instantaneous jumps between regimes
   - Multiple modes in resulting distributions
   - Heavy tails from mixture of exponentials

5. **Physical interpretability**:
   - States can represent ecological conditions
   - Transitions model environmental changes
   - Stationary distribution gives long-run population behavior

### Implementation: Auxiliary Chain for $N(t)$

```python
import numpy as np
from phasic import Graph

def build_coalescent_with_auxiliary(n_samples, K_aux, N_values, Q_aux, epoch_boundaries):
    """
    Build coalescent graph with auxiliary chain governing N(t).

    Parameters
    ----------
    n_samples : int
        Number of samples
    K_aux : int
        Number of auxiliary states
    N_values : array, shape (K_aux,)
        Population size for each auxiliary state
    Q_aux : array, shape (K_aux, K_aux)
        Sub-intensity matrix for auxiliary chain
    epoch_boundaries : array
        Time boundaries for epochs

    Returns
    -------
    graph : Graph
        Combined coalescent + auxiliary state graph
    """
    # State: (n_lineages, aux_state)
    # Transitions:
    #   1. Coalescence: (n, z) → (n-1, z) at rate n(n-1)/(2*N_z)
    #   2. Aux change: (n, z) → (n, z') at rate Q_aux[z, z']

    def combined_callback(state):
        n = state[0]  # Number of lineages
        z = state[1]  # Auxiliary state (0-indexed)

        if n <= 1:
            return []  # Absorbed

        transitions = []

        # Coalescence transitions
        coal_rate = n * (n - 1) / (2 * N_values[z])
        transitions.append((np.array([n - 1, z]), coal_rate))

        # Auxiliary state transitions
        for z_new in range(K_aux):
            if z_new != z and Q_aux[z, z_new] > 0:
                transitions.append((np.array([n, z_new]), Q_aux[z, z_new]))

        return transitions

    # Initial state: n lineages, weighted over auxiliary states
    # (Can start from stationary distribution of auxiliary chain)

    graph = Graph(
        state_length=2,
        callback=combined_callback,
        parameterized=True,
        nr_samples=n_samples
    )

    return graph


def fit_auxiliary_model(observations, K_aux=5, n_epochs=25):
    """
    Fit auxiliary chain model to observations.

    Parameters
    ----------
    observations : dict
        {'tmrca': array, 'singletons': array, ...}
    K_aux : int
        Number of auxiliary states
    n_epochs : int
        Number of time epochs for discretization

    Returns
    -------
    result : dict
        Fitted parameters and diagnostics
    """
    from phasic import SVGD

    # Parameterize: [N_1, ..., N_K, q_12, q_13, ..., q_KK-1]
    # where Q[i,j] = q_ij for i ≠ j, Q[i,i] = -sum_j Q[i,j]

    n_N_params = K_aux
    n_Q_params = K_aux * (K_aux - 1)
    theta_dim = n_N_params + n_Q_params

    def objective(theta):
        # Unpack parameters
        N_values = theta[:n_N_params]
        Q_flat = theta[n_N_params:]

        # Build Q matrix
        Q_aux = np.zeros((K_aux, K_aux))
        idx = 0
        for i in range(K_aux):
            for j in range(K_aux):
                if i != j:
                    Q_aux[i, j] = Q_flat[idx]
                    idx += 1
            Q_aux[i, i] = -Q_aux[i, :].sum()

        # Build graph and compute likelihood
        graph = build_coalescent_with_auxiliary(...)

        # ... likelihood computation ...

        return log_lik

    # Run SVGD
    svgd = SVGD(
        model=objective,
        theta_dim=theta_dim,
        n_particles=100,
        n_iterations=1000
    )

    return svgd.fit()
```

### Demonstration: Fitting Various Population Histories

The following figures demonstrate phase-type models fitting complex population histories with 25 parameters.

![Phase-Type Fits](python/phase_type_25_param_demo.png)

*Figure 1: Phase-type distribution fits to various population histories. Top row: True N(t) (black), spline fit (green), and phase-type fit (blue). Bottom row: TMRCA density histogram (gray) with kernel density estimate (black) and phase-type model (blue). From left to right: smooth decline, bottleneck, oscillating, step function.*

Key observations:
1. **Smooth histories**: Both splines and phase-type perform well
2. **Bottlenecks**: Phase-type captures sharp transitions; splines smooth them
3. **Oscillations**: Phase-type with multiple auxiliary states captures multi-modal structure
4. **Step functions**: Phase-type exactly matches; splines introduce artificial smoothing

### Detailed Comparison: Bottleneck Scenario

![Spline vs Phase-Type](python/phase_type_vs_spline_comparison.png)

*Figure 2: Comparison of spline and phase-type approximations for a bottleneck scenario. Left: True population history with sharp bottleneck (red shading). Middle: Spline approximation smooths out the bottleneck. Right: Phase-type approximation captures the discontinuity.*

The bottleneck scenario highlights the fundamental difference:
- **Splines** enforce smoothness, which biases the estimate towards gradual changes
- **Phase-type** naturally handles discontinuities through regime switching

### Auxiliary Chain Structures

![Auxiliary Chains](python/auxiliary_chain_demo.png)

*Figure 3: Different auxiliary chain structures for parameterizing N(t). Left: Two-state chain (boom-bust). Middle: Birth-death chain (gradual changes). Right: Coxian chain (sequential progression).*

The choice of auxiliary chain structure encodes prior knowledge about the population dynamics:
- **Two-state**: Population alternates between two regimes (expansion/contraction)
- **Birth-death**: Gradual transitions with possible reversals
- **Coxian**: Monotonic progression through stages (e.g., successive bottlenecks)

---

## References

- [Hansen (1982)](https://doi.org/10.2307/1912775) - Large sample properties of GMM estimators
- [White (1982)](https://doi.org/10.2307/1912526) - MLE under misspecification
- [Newey and McFadden (1994)](https://doi.org/10.1016/S1573-4412(05)80005-4) - Large sample estimation and hypothesis testing
- [Owen (2001)](https://www.routledge.com/Empirical-Likelihood/Owen/p/book/9781584880714) - Empirical Likelihood
- [Beaumont et al. (2002)](https://doi.org/10.1093/genetics/162.4.2025) - ABC in population genetics
- [Hobolth et al. (2019)](https://doi.org/10.1214/18-STS663) - Phase-type distributions in population genetics
