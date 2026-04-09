# Phasic Notation Standard

**Version:** 1.0  
**Status:** Draft  
**Scope:** All phasic documentation, supplementary materials, and future papers  

## 1. Purpose and Scope

This document defines the standard mathematical, graph-theoretic, and algorithmic notation for all documentation produced for the phasic project. It governs:

- Exhaustive mathematical and algorithmic documentation on the phasic documentation website
- Sections extracted for inclusion in papers or supplementary materials
- Algorithm pseudocode in all contexts

The notation is designed to be acceptable to both the **applied probability / mathematical statistics** community and the **computer science / graph algorithms** community. It draws on conventions established in:

- Bladt & Nielsen (2017), *Matrix-Exponential Distributions in Applied Probability* (the standard PH textbook)
- Røikjer, Hobolth & Munch (2022), *Graph-based algorithms for phase-type distributions* (the phasic paper)
- Hobolth, Rivas-González, Bladt & Futschik (2024), *Phase-type distributions in mathematical population genetics*
- Neuts (1981), *Matrix-Geometric Solutions in Stochastic Models*
- Cormen, Leiserson, Rivest & Stein (CLRS), *Introduction to Algorithms* (for pseudocode and graph conventions)

Where these sources conflict, this standard specifies the resolution.

## 2. Typographic Conventions

### 2.1 General Rules

| Entity | Convention | Examples | Rendering |
|--------|-----------|----------|-----------|
| Matrices | Bold uppercase | **S**, **T**, **U**, **I**, **R**, **K** | `\mathbf{S}`, `\boldsymbol{\Lambda}` |
| Vectors | Bold lowercase | **α**, **t**, **e**, **r**, **s** | `\boldsymbol{\alpha}`, `\mathbf{t}` |
| Scalars | Non-bold italic | *p*, *t*, *λ*, *n* | `p`, `t`, `\lambda`, `n` |
| Random variables | Non-bold uppercase italic | *X*, *Y*, *τ*, *H*, *L* | `X`, `Y`, `\tau`, `H`, `L` |
| Realizations / observed values | Non-bold lowercase italic | *x*, *y*, *s*, *t* | `x`, `y`, `s`, `t` |
| Sets | Non-bold uppercase italic | *V*, *E*, *N* | `V`, `E`, `\mathbb{N}` |
| Functions | Non-bold italic or roman | *f*, *w*, children, parents | `f`, `w`, `\operatorname{children}` |
| Operators / named functions | Upright roman | exp, log, diag, Pr | `\exp`, `\log`, `\operatorname{diag}`, `\Pr` |
| Named constants / enum values / operation types | Monospace (typewriter) | Const, Add, Mul, Div, Inv, Dot, Param, Sub, Sum | `\texttt{Const}`, `\texttt{Add}`, `\texttt{Mul}` |

### 2.2 LaTeX Implementation

Use `\mathbf{}` for roman bold (latin matrices and vectors: **S**, **T**, **e**, **t**) and `\boldsymbol{}` for bold Greek (**α**, **π**, **Λ**). This ensures correct rendering across journal LaTeX classes.

### 2.3 Named Constants and Operation Types in Math Mode

When named constants, enumeration values, or operation types appear inside mathematical expressions (e.g., expression tree node types, trace operation types), render them in **monospace** using `\texttt{}`:

$$\mathcal{E} = \texttt{Add}(\texttt{Param}(1), \texttt{Const}(3.0))$$

**Do not use `\textsc{}`** (LaTeX small caps). The `\textsc` command is not supported by MathJax and will fail to render in the Quarto documentation site and in any markdown context using MathJax or KaTeX. The `\texttt{}` command is supported by MathJax, KaTeX, and all standard LaTeX document classes.

**Rationale:** Monospace visually distinguishes named constants from mathematical variables (which are italic) and from operators (which are upright roman). It also mirrors the convention that these names correspond to code-level enum values (e.g., `PTD_EXPR_CONST`, `OpType.DOT`).

**In pseudocode blocks** (outside math mode), named constants may appear as plain PascalCase text (e.g., `Const`, `Add`) without any special formatting, since pseudocode already uses non-bold italic for all variables (Section 8.1, rule 4).

### 2.4 Markdown / Documentation Site

In markdown contexts where LaTeX math mode is available (e.g., Quarto, MathJax), use the same LaTeX commands inside `$...$` delimiters. In plain-text contexts, denote vectors and matrices in **bold** using markdown bold syntax and scalars in *italic*.

## 3. Phase-Type Distribution Notation

### 3.1 Continuous Phase-Type Distribution (CPH)

A **continuous phase-type distribution** is the time until absorption in a continuous-time Markov jump process on the finite state space $\{1, 2, \ldots, p, p+1\}$, where states $1, \ldots, p$ are transient and state $p+1$ is absorbing.

| Symbol | Name | Definition |
|--------|------|------------|
| $p$ | Number of transient states | A positive integer |
| $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_p)$ | Initial distribution vector | Row vector; $\alpha_i = \Pr(X_0 = i)$; may be sub-stochastic: $\sum_i \alpha_i \leq 1$ |
| $\alpha_0 = 1 - \sum_{i=1}^{p} \alpha_i$ | Defect | Probability of starting in the absorbing state |
| $\mathbf{S}$ | Sub-intensity matrix | $p \times p$ matrix of transition rates between transient states; off-diagonal entries $s_{ij} \geq 0$ for $i \neq j$; diagonal entries $s_{ii} < 0$; row sums non-positive |
| $\mathbf{s} = -\mathbf{S}\mathbf{e}$ | Exit rate vector | Column vector of size $p$; entry $s_i$ is the rate of absorption from state $i$ |
| $\mathbf{e}$ | Column vector of ones | $\mathbf{e} = (1, 1, \ldots, 1)^\top$ of appropriate dimension |
| $\boldsymbol{\Lambda}$ | Full intensity matrix | $(p+1) \times (p+1)$ matrix: $\boldsymbol{\Lambda} = \begin{pmatrix} \mathbf{S} & \mathbf{s} \\ \mathbf{0} & 0 \end{pmatrix}$ |
| $\mathbf{U} = (-\mathbf{S})^{-1}$ | Green matrix | Entry $U_{ij}$ is the expected total time spent in state $j$ given start in state $i$ |
| $\tau \sim \operatorname{PH}_p(\boldsymbol{\alpha}, \mathbf{S})$ | Phase-type distributed r.v. | Time until absorption; $p$ is the order |

**Probability density function:**
$$f_\tau(t) = \boldsymbol{\alpha} e^{\mathbf{S}t} \mathbf{s}, \quad t \geq 0.$$

**Cumulative distribution function:**
$$F_\tau(t) = 1 - \boldsymbol{\alpha} e^{\mathbf{S}t} \mathbf{e}, \quad t \geq 0.$$

**Moments:**
$$\mathbb{E}[\tau^n] = n! \, \boldsymbol{\alpha} (-\mathbf{S})^{-n} \mathbf{e}.$$

**Laplace transform:**
$$\mathcal{L}_\tau(u) = \boldsymbol{\alpha}(u\mathbf{I} - \mathbf{S})^{-1} \mathbf{s}, \quad u \geq 0.$$

### 3.2 Discrete Phase-Type Distribution (DPH)

A **discrete phase-type distribution** is the number of jumps until absorption in a discrete-time Markov chain on $\{1, \ldots, p, p+1\}$.

| Symbol | Name | Definition |
|--------|------|------------|
| $\mathbf{T}$ | Sub-transition matrix | $p \times p$ matrix of transition probabilities between transient states; entries $t_{ij} \geq 0$; row sums $\leq 1$ |
| $\mathbf{t} = \mathbf{e} - \mathbf{T}\mathbf{e}$ | Exit probability vector | Column vector; $t_i = 1 - \sum_j T_{ij}$ |
| $\boldsymbol{\pi}$ or $\boldsymbol{\alpha}$ | Initial distribution | Row vector; use $\boldsymbol{\alpha}$ when both CPH and DPH appear together |
| $\tau \sim \operatorname{DPH}_p(\boldsymbol{\alpha}, \mathbf{T})$ | DPH distributed r.v. | Number of jumps until absorption |

**Probability mass function:**
$$\Pr(\tau = k) = \boldsymbol{\alpha} \mathbf{T}^{k-1} \mathbf{t}, \quad k = 1, 2, \ldots$$

**Moments:**
$$\mathbb{E}[\tau] = \boldsymbol{\alpha}(\mathbf{I} - \mathbf{T})^{-1}\mathbf{e}.$$

> **Convention note:** We use **S** for the continuous sub-intensity matrix and **T** for the discrete sub-transition matrix. When only one type is discussed and context is clear, **T** may appear for either. When both appear in the same document, the S/T distinction is mandatory.

### 3.3 Reward-Transformed Phase-Type Distributions

| Symbol | Name | Definition |
|--------|------|------------|
| $\mathbf{r} = (r_1, \ldots, r_p)^\top$ | Reward vector | Non-negative rewards assigned to transient states |
| $\triangle(\mathbf{r})$ | Diagonal reward matrix | $p \times p$ diagonal matrix with $\mathbf{r}$ on the diagonal |
| $\tilde{\tau} = \int_0^{\tau} r(X_t) \, dt$ | Reward-transformed r.v. | Accumulated reward until absorption (CPH) |
| $\tilde{\tau} = \sum_{t=0}^{\tau} r(X_t)$ | Reward-transformed r.v. | Accumulated reward until absorption (DPH) |

If all rewards are strictly positive, then $\tilde{\tau} \sim \operatorname{PH}(\boldsymbol{\alpha}, \triangle(\mathbf{r})^{-1}\mathbf{S})$.

**Moments of reward-transformed distributions:**
$$\mathbb{E}[\tilde{\tau}^n] = n! \, \boldsymbol{\alpha} \left( (-\mathbf{S})^{-1} \triangle(\mathbf{r}) \right)^n \mathbf{e}.$$

### 3.4 Multivariate Phase-Type Distributions (MPH)

| Symbol | Name | Definition |
|--------|------|------------|
| $\mathbf{R}$ | Reward matrix | $p \times m$ matrix; column $j$ is the reward vector for marginal $j$ |
| $\mathbf{Y} = (Y_1, \ldots, Y_m)$ | MPH distributed r.v. | $Y_j = \int_0^\tau r_j(X_t) \, dt$ |
| $\mathbf{Y} \sim \operatorname{MPH}(\boldsymbol{\alpha}, \mathbf{S}, \mathbf{R})$ | MPH notation | Multivariate phase-type distribution |

**Cross-moments:**
$$\mathbb{E}[Y_i Y_j] = \boldsymbol{\alpha} \mathbf{U} \triangle(\mathbf{R}_{\cdot i}) \mathbf{U} \triangle(\mathbf{R}_{\cdot j}) \mathbf{e} + \boldsymbol{\alpha} \mathbf{U} \triangle(\mathbf{R}_{\cdot j}) \mathbf{U} \triangle(\mathbf{R}_{\cdot i}) \mathbf{e},$$
where $\mathbf{R}_{\cdot j}$ denotes column $j$ of $\mathbf{R}$.

## 4. Graph Notation

### 4.1 Directed Weighted Graph

| Symbol | Name | Definition |
|--------|------|------------|
| $G = (V, E)$ | Directed graph | $V$ is the set of vertices, $E \subseteq V \times V$ is the set of directed edges |
| $(v \to z) \in E$ | Directed edge | An edge from vertex $v$ to vertex $z$ |
| $W \colon E \to \mathbb{R}$ | Weight function | Assigns a real-valued weight to each edge |
| $w(v \to z)$ | Edge weight | The weight of edge $(v \to z)$ |
| $\lambda_v = \sum_{z \in \operatorname{children}(v)} w(v \to z)$ | Total outgoing rate | Sum of weights of all outgoing edges of $v$ |
| $\operatorname{children}(v)$ | Children of $v$ | $\{z \in V : (v \to z) \in E\}$ |
| $\operatorname{parents}(v)$ | Parents of $v$ | $\{u \in V : (u \to v) \in E\}$ |
| $\lvert V \rvert$ | Number of vertices | Cardinality of the vertex set |
| $\lvert E \rvert$ | Number of edges | Cardinality of the edge set |

### 4.2 Phase-Type Graph Representation

A phase-type distribution $\operatorname{PH}_p(\boldsymbol{\alpha}, \mathbf{S})$ is represented as a weighted directed graph $G = (V, E)$ where:

| Symbol | Name | Definition |
|--------|------|------------|
| $v_0$ | Starting vertex | Designated vertex representing the initial distribution; $w(v_0 \to v_i) = \alpha_i$ and $\lambda_{v_0} = 1$; assigned reward zero |
| $v_1, \ldots, v_p$ | Transient vertices | One vertex per transient state; edge weights correspond to off-diagonal entries of $\mathbf{S}$ |
| Absorbing vertices | Vertices with no outgoing edges | Represent absorbing states |
| $r_v$ or $r(v)$ | Vertex reward | Non-negative scalar associated with vertex $v$ |
| $x_v$ | Vertex scalar | Expected waiting time at vertex $v$; after normalization, $x_v = \lambda_v^{-1}$ |

> **Convention note on the starting vertex:** We use $v_0$ (not $S$) for the starting vertex, because $\mathbf{S}$ is reserved for the sub-intensity matrix. The published Røikjer et al. (2022) paper uses $S$ for the starting vertex; this standard resolves that overloading.

### 4.3 Graph Transformations

| Notation | Meaning |
|----------|---------|
| $G' = (V', E')$ | Transformed graph; prime denotes result of a transformation |
| $G \leftarrow G'$ | In-place update of graph $G$ |
| $G^{(i)}$ | Graph after processing vertex with index $i$ |
| $w'(v \to z) = w(v \to z) / \lambda_v$ | Normalized edge weight (transition probability) |

### 4.4 Parameterized Edges

When edge weights depend on a parameter vector $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_k)$, each edge carries a coefficient vector $\mathbf{c} = (c_1, \ldots, c_k)$. The weight is computed according to the **weight mode**:

| Mode | Weight computation | Use case |
|------|-------------------|----------|
| `linear` (default) | $w = \sum_{j} c_j \theta_j$ | Additive parameterization |
| `log` | $w = \prod_{j} (c_j \theta_j)$ (computed in log-space) | Multiplicative parameterization |
| `callback` | $w = \phi(\boldsymbol{\theta}, \mathbf{c})$ for arbitrary function $\phi$ | Non-linear parameterization |

## 5. Markov Chain and Stochastic Process Notation

| Symbol | Name | Definition |
|--------|------|------------|
| $\{X_t\}_{t \geq 0}$ | Continuous-time Markov jump process | State at time $t$ is $X_t$ |
| $\{X_t\}_{t \in \mathbb{N}_0}$ | Discrete-time Markov chain | State at step $t$ is $X_t$ |
| $\tau = \inf\{t : X_t = p+1\}$ | Absorption time | First time the process enters the absorbing state |
| $s_{ij}$ | Transition rate (CPH) | Entry $(i,j)$ of $\mathbf{S}$; for $i \neq j$, rate of jumping from $i$ to $j$ |
| $t_{ij}$ | Transition probability (DPH) | Entry $(i,j)$ of $\mathbf{T}$; probability of jumping from $i$ to $j$ |
| $q_{ij} = s_{ij} / \lambda_i$ | Embedded chain probability | Transition probability of the embedded discrete-time chain |
| $Z_i$ | Occupation time | Total time spent in state $i$ before absorption |
| $N_{ij}$ | Transition count | Number of transitions from state $i$ to state $j$ |
| $B_i$ | Starting count | Number of processes initiating in state $i$ (in a sample) |

## 6. Probability and Statistics Notation

| Symbol | Name | Usage |
|--------|------|-------|
| $\mathbb{E}[\cdot]$ | Expectation | Blackboard-bold E |
| $\operatorname{Var}(\cdot)$ | Variance | Roman upright |
| $\operatorname{Cov}(\cdot, \cdot)$ | Covariance | Roman upright |
| $\Pr(\cdot)$ | Probability | Roman upright |
| $\sim$ | Distributed as | $X \sim \operatorname{PH}(\boldsymbol{\alpha}, \mathbf{S})$ |
| $\mid$ | Conditional on | $\mathbb{E}[X \mid Y]$ |
| $f_X(t)$ or $f(t)$ | Probability density function | Subscript specifies the r.v. when ambiguous |
| $F_X(t)$ or $F(t)$ | Cumulative distribution function | |
| $\theta$ | Generic parameter | Scalar; use $\boldsymbol{\theta}$ for parameter vector |
| $\hat{\theta}$ | Estimator | Hat notation for point estimates |
| $\ell(\boldsymbol{\theta}; \mathbf{y})$ | Log-likelihood | Lowercase script $\ell$; data $\mathbf{y}$; parameter $\boldsymbol{\theta}$ |
| $L(\boldsymbol{\theta}; \mathbf{y})$ | Likelihood | Uppercase $L$ |
| $\mathcal{L}_\tau(u)$ | Laplace transform | Calligraphic $\mathcal{L}$ with subscript for the r.v. |

### 6.1 Standard Distributions

| Distribution | Notation |
|-------------|----------|
| Exponential with rate $\lambda$ | $\operatorname{Exp}(\lambda)$ |
| Poisson with mean $\mu$ | $\operatorname{Poisson}(\mu)$ or $\operatorname{Po}(\mu)$ |
| Geometric with success prob. $p$ | $\operatorname{Geo}(p)$; $\Pr(X = k) = p^k(1-p)$, $k = 0, 1, \ldots$ |
| Binomial | $\operatorname{Binom}(n, p)$ |
| Erlang | $\operatorname{Erlang}(k, \lambda)$ |

## 7. Algorithmic Notation

### 7.1 Specific Mathematical Notation in Algorithms

| Symbol | Name | Definition |
|--------|------|------------|
| $\mathbf{e}_i$ | Standard basis vector | Column vector with 1 in position $i$ and 0 elsewhere |
| $\mathbf{I}$ | Identity matrix | $p \times p$ identity; subscript $\mathbf{I}_p$ when dimension must be explicit |
| $e^{\mathbf{S}t}$ | Matrix exponential | $\sum_{k=0}^{\infty} \frac{(\mathbf{S}t)^k}{k!}$ |
| $\mathbf{A}^\top$ | Transpose | Superscript $\top$ (upright T) |
| $\mathbf{A}^{-1}$ | Matrix inverse | |
| $\lVert \cdot \rVert$ | Norm | Specify which norm on first use |
| $\mathbf{0}$ | Zero vector or matrix | Dimension inferred from context |
| $\mathbb{1}_{\{A\}}$ or $\mathbf{1}_A$ | Indicator function | Equals 1 when condition $A$ holds, 0 otherwise |

### 7.2 Asymptotic Notation

| Symbol | Meaning |
|--------|---------|
| $O(f(n))$ | Asymptotic upper bound |
| $\Omega(f(n))$ | Asymptotic lower bound |
| $\Theta(f(n))$ | Asymptotically tight bound |

## 8. Algorithm Pseudocode Conventions

### 8.1 Formatting Rules

1. **Line numbers:** Arabic numerals followed by a colon (e.g., `1:`, `2:`), left-aligned.
2. **Keywords:** Bold lowercase: **function**, **for**, **while**, **if**, **then**, **else**, **end**, **return**, **do**.
3. **Function names:** PascalCase (e.g., `GenerateStateSpace`, `RewardTransformVertex`).
4. **Variables:** Italic (e.g., *v*, *z*, *unvisited*). No bold distinction for vectors vs. scalars inside pseudocode blocks.
5. **Comments:** Preceded by the triangleright symbol ($\triangleright$) on the same line.
6. **Indentation:** Two spaces per nesting level.
7. **Set operations:** Use standard mathematical notation: $\cup$, $\setminus$, $\in$, $\emptyset$, $\subseteq$.
8. **Assignment:** Left arrow $\leftarrow$ (e.g., $x \leftarrow x + 1$).
9. **Algorithm header:** Each algorithm has a number and a descriptive title.
10. **Input/output:** State preconditions and postconditions as numbered preamble lines before the **function** keyword.
11. **Return values:** Explicitly stated with **return**.

### 8.2 Example Template

```
Algorithm N: Descriptive Title
1: Let ... describe preconditions
2:
3: function FunctionName(parameters)
4:   initialization
5:   for v ∈ V do
6:     body                          ▷ Comment explaining this step
7:   end for
8:   return result
9: end function
```

### 8.3 Rules for Mathematical Expressions in Pseudocode

- Inline math expressions follow the same conventions as in the text (subscripts, Greek letters, etc.)
- When referencing matrices or vectors that are bold in the surrounding text, they appear non-bold and italic inside pseudocode, relying on context (the variable name and surrounding algorithm) for disambiguation
- Use $\leftarrow$ for assignment, $=$ only for equality tests and definitions

## 9. Numbering and Cross-Referencing

### 9.1 Definitions, Theorems, Lemmas

Use a single counter per section:

- **Definition N.M** (e.g., Definition 2.1)
- **Theorem N.M**
- **Lemma N.M**
- **Proposition N.M**
- **Corollary N.M**
- **Proof.** Begins with "Proof." in italic, ends with $\square$.

### 9.2 Algorithms

Algorithms are numbered with a separate counter: **Algorithm 1**, **Algorithm 2**, etc.

### 9.3 Equations

Equations are numbered sequentially within each document section: (1), (2), etc. Only number equations that are referenced elsewhere.

## 10. Documentation Structure Conventions

### 10.1 Narrative Definition-by-Definition Format

All mathematical documentation must follow a **narrative definition-by-definition** structure:

1. **Introduce context** in prose before any formal definition.
2. **State definitions** formally using the Definition environment.
3. **Follow each definition** with:
   - An intuitive explanation of what the definition captures
   - The relationship to prior definitions
   - A concrete example when the concept is non-trivial
4. **State results** (theorems, lemmas, propositions) formally.
5. **Provide proofs** inline (not deferred to appendices) for completeness.
6. **State algorithms** with:
   - A prose description of the algorithm's purpose and strategy
   - The formal pseudocode
   - A complexity analysis
   - A correctness argument or proof

### 10.2 Symbol Index

Every documentation page that introduces notation must include a **Symbol Index** section at the end, listing all symbols introduced on that page in alphabetical order with brief definitions and the definition/equation number where they first appear.

## 11. Conflict Resolutions

This section records explicit decisions where conventions from different fields conflict.

| Conflict | Resolution | Rationale |
|----------|-----------|-----------|
| **S** for sub-intensity matrix vs. **S** for starting vertex | Use **S** for the matrix; use $v_0$ for the starting vertex | **S** for sub-intensity is the textbook standard (Bladt & Nielsen 2017) and matches the published phasic paper's matrix notation |
| **T** for sub-transition matrix vs. **T** for transpose | Use **T** for sub-transition; use superscript ${}^\top$ for transpose | The upright $\top$ vs italic $T$ distinction resolves ambiguity |
| **e** for ones vector vs. *e* for Euler's number vs. $\mathbf{e}_i$ for basis vectors | **e** (bold) for ones vector; $e$ or $\exp(\cdot)$ for Euler's number; $\mathbf{e}_i$ (bold with subscript) for basis vectors | Bold/non-bold disambiguates; consistent with Bladt & Nielsen and Hobolth et al. |
| $\lambda_v$ for total outgoing rate vs. $\lambda$ for eigenvalue | $\lambda_v$ (subscripted) for vertex rate; $\lambda$ (unsubscripted) for eigenvalue or generic rate | Subscript disambiguates |
| **π** for initial distribution vs. **π** for stationary distribution | Use **α** for initial distribution; reserve **π** for stationary distributions | Avoids ambiguity; consistent with the majority of PH literature |
| Bold vectors in text vs. non-bold in pseudocode | Bold in mathematical text and equations; non-bold italic in pseudocode | The context (equation vs. algorithm block) disambiguates; standard practice in Springer publications |

## 12. Symbol Quick Reference

### Greek Letters

| Symbol | Bold? | Meaning |
|--------|-------|---------|
| $\boldsymbol{\alpha}$ | yes | Initial distribution vector |
| $\alpha_0$ | no | Defect (probability of starting in absorbing state) |
| $\alpha_i$ | no | Probability of starting in state $i$ |
| $\boldsymbol{\theta}$ | yes | Parameter vector |
| $\theta_j$ | no | Individual parameter |
| $\lambda_v$ | no | Total outgoing rate of vertex $v$ |
| $\lambda_i$ | no | Rate parameter (e.g., coalescent rate $\binom{i}{2}$) |
| $\boldsymbol{\Lambda}$ | yes | Full intensity/transition matrix (including absorbing state) |
| $\boldsymbol{\pi}$ | yes | Stationary distribution (not used for initial distribution) |
| $\tau$ | no | Absorption time (random variable) |
| $\tilde{\tau}$ | no | Reward-transformed absorption time |

### Latin Letters

| Symbol | Bold? | Meaning |
|--------|-------|---------|
| $\mathbf{e}$ | yes | Column vector of ones |
| $\mathbf{e}_i$ | yes | Standard basis vector (1 in position $i$) |
| $E$ | no | Set of edges |
| $G = (V, E)$ | no | Directed graph |
| $\mathbf{I}$ | yes | Identity matrix |
| $\mathbf{R}$ | yes | Reward matrix ($p \times m$) |
| $\mathbf{r}$ | yes | Reward vector |
| $\mathbf{S}$ | yes | Sub-intensity matrix (CPH) |
| $\mathbf{s}$ | yes | Exit rate vector ($= -\mathbf{S}\mathbf{e}$) |
| $\mathbf{T}$ | yes | Sub-transition matrix (DPH) |
| $\mathbf{t}$ | yes | Exit probability vector ($= \mathbf{e} - \mathbf{T}\mathbf{e}$) |
| $\mathbf{U}$ | yes | Green matrix ($= (-\mathbf{S})^{-1}$) |
| $V$ | no | Set of vertices |
| $v_0$ | no | Starting vertex |
| $W$ | no | Weight function $W: E \to \mathbb{R}$ |
| $w(v \to z)$ | no | Weight of a specific edge |
| $x_v$ | no | Vertex scalar (expected waiting time) |

## 13. Updating This Standard

This section defines the mandatory procedure for changing any notation in this standard. The goal is to ensure that no change silently breaks mathematical correctness or introduces inconsistency across documents.

### 13.1 When to Update

A notation change is required when:

- A new symbol is needed that collides with an existing one
- A reviewer or co-author requests a different convention for a submission
- A new concept (e.g., a new distribution type, a new graph operation) has no existing notation entry
- An error or ambiguity is discovered in the current standard

A notation change is **not** required when:

- A document introduces a local shorthand that is defined at point of use and does not conflict with this standard (e.g., "let $A = \mathbf{U}\triangle(\mathbf{r})$ for brevity")
- A journal's house style differs only in rendering (e.g., `\bm` vs `\mathbf`) — adapt rendering at submission time, not the standard

### 13.2 Change Proposal

Before modifying `notation_standard.md`, write a **change proposal** as a short markdown section (in a scratch file, conversation, or issue) containing:

1. **What:** The exact symbol(s) being added, removed, or changed. Show old → new.
2. **Why:** The reason — conflict, new concept, reviewer request, etc.
3. **Scope:** Which existing documents use the affected symbol(s).

### 13.3 Impact Analysis (Pre-Flight Check)

Before any change is applied to documents, verify that it does not break mathematical logic or algorithmic correctness:

**Step 1 — Inventory all uses.** Search every document that this standard governs for every occurrence of the symbol being changed:

```bash
# Search docs site, papers, and supplementary materials
# Use exact LaTeX patterns to avoid false positives
rg --type md --type tex --type qmd -n 'EXACT_LATEX_PATTERN' docs/ papers/ supplementary/

# Examples:
rg -n '\\mathbf\{S\}' docs/ papers/
rg -n 'v_0' docs/ papers/
rg -n '\\boldsymbol\{\\alpha\}' docs/ papers/
```

Record the file paths and line numbers of every hit. This is the **affected set**.

**Step 2 — Check for semantic collisions.** For each file in the affected set, verify:

- The new symbol does not already appear in that file with a different meaning
- The new symbol does not collide with a variable name in any algorithm pseudocode block in that file
- If the change is a rename (A → B), confirm that B is not already in use anywhere in the affected set

```bash
# Check if the NEW symbol already appears somewhere with a different meaning
rg -n 'NEW_LATEX_PATTERN' docs/ papers/
```

**Step 3 — Verify mathematical identities.** For every equation in the affected set that uses the changed symbol, manually verify that the substitution preserves:

- Dimensional consistency (matrix × vector = vector, not scalar × matrix)
- The equation's mathematical meaning (e.g., if renaming **S** → **Q**, confirm that $\mathbf{s} = -\mathbf{S}\mathbf{e}$ becomes $\mathbf{q} = -\mathbf{Q}\mathbf{e}$ and that all downstream uses of **s** are also updated)
- Cross-references: any equation that references another equation by number still makes sense after the substitution

Write down each verified equation with a checkmark. This is the **verification log**.

**Step 4 — Verify algorithm pseudocode.** For every algorithm in the affected set:

- Trace the changed symbol through the algorithm line by line
- Confirm that input/output types are preserved
- Confirm that loop invariants and postconditions still hold
- If the algorithm references an equation, confirm the equation was verified in Step 3

**Step 5 — Check the Symbol Quick Reference (Section 12).** Confirm the old entry will be removed and the new entry added, with correct bold/non-bold and meaning.

### 13.4 Applying Changes Across Documents

Once the impact analysis passes, apply changes in this exact order:

**Phase 1 — Update `notation_standard.md` first.**

Edit the standard itself:
- Update the relevant table(s) in Sections 3–7
- Update the Symbol Quick Reference (Section 12)
- Update the Conflict Resolutions table (Section 11) if the change resolves or creates a conflict
- Add an entry to the Revision History (Section 14)

**Phase 2 — Search and replace across all governed documents.**

Use targeted search-and-replace, not bulk find-replace. Process one file at a time:

```bash
# List all files containing the old pattern
rg -l 'OLD_PATTERN' docs/ papers/ supplementary/

# For each file, review matches in context before replacing
rg -n -C 2 'OLD_PATTERN' path/to/file.md

# Apply replacement only after reviewing context
# Use sed, editor find-replace, or Claude's Edit tool — one file at a time
```

**Rules for search-and-replace:**

- Never do a blind global replace across all files in one command. Always review context per file.
- Replace LaTeX math-mode occurrences and plain-text occurrences separately (they may have different patterns).
- After replacing in a file, re-read the surrounding paragraph to confirm it still reads correctly.
- If a file is a published paper (already submitted/accepted), do **not** modify it. Instead, add a note in the Revision History recording the divergence.

**Phase 3 — Verify the replacement.**

After all files are updated:

```bash
# Confirm no old-pattern stragglers remain
rg -n 'OLD_PATTERN' docs/ papers/ supplementary/
# Should return zero results (except in published/frozen papers)

# Confirm the new pattern appears where expected
rg -c 'NEW_PATTERN' docs/ papers/ supplementary/
```

**Phase 4 — Build and inspect.**

If the docs site uses Quarto or another build system:

```bash
pixi run docs-build   # or equivalent
```

Inspect any pages that were modified. Confirm that:
- Math renders correctly (no broken LaTeX)
- No symbol appears undefined
- Cross-references still resolve

### 13.5 Special Case: Journal-Specific Overrides

When submitting to a journal that requires different notation (e.g., a journal that mandates $\pi$ instead of $\boldsymbol{\alpha}$ for initial distributions):

1. Create a **submission overlay file** (e.g., `papers/journal_name/notation_overrides.md`) listing every deviation from this standard.
2. Apply the overrides only in the submission copy, not in the canonical documentation.
3. Record the override in the Revision History of this standard, noting which paper and journal.
4. After acceptance, do **not** backport journal-specific notation into this standard unless the change is genuinely better.

### 13.6 Governance

- `notation_standard.md` is the single source of truth. If a document disagrees with this standard, the document is wrong unless it is a frozen published paper.
- Any contributor (human or AI) producing mathematical documentation for phasic must load this standard at the start of the task.
- Changes to this standard require the full procedure in Sections 13.2–13.4. Skipping the impact analysis is not permitted.

## 14. Agent Team for Documentation Production

This section specifies the team of specialized agents that together produce exhaustive, mathematically correct, and algorithmically verified documentation from the phasic codebase. The codebase is the sole input; the agents must extract, formalize, prove, and verify everything from source code, tests, and existing technical documents.

### 14.0 Foundational Principle: Code Correctness Is Not Assumed

**No agent may assume the source code is correct.** The agents document what the code does and attempt to prove that what it does is mathematically sound. If an agent cannot construct a valid proof, or discovers that the code's behavior contradicts established mathematical results, the agent must halt and report to the human — not fabricate a proof, not silently weaken the theorem statement to match the code, and not modify the code.

**Specific obligations by agent:**

| Agent | What to do when code appears wrong |
|-------|-----------------------------------|
| Agent 1 (Codebase Analyst) | If the code's logic is internally inconsistent (e.g., a loop invariant is violated, a claimed optimization changes behavior), report the inconsistency in the extraction report with `SUSPECTED_CODE_ISSUE` flag, citing the specific lines. Continue extracting what the code actually does. |
| Agent 2 (Math Formalization) | If a proof of correctness cannot be constructed — i.e., the algorithm as extracted does not satisfy the theorem it should — **do not weaken the theorem to match the code**. Instead: (1) state the theorem as it should be (the mathematically correct version), (2) state what the code actually computes, (3) identify the gap, (4) issue a `SOURCE_CODE_BUG` report to the human per Section 14.2.7. The documentation draft should include both the correct theorem and the discrepancy note, so the human can see exactly what is wrong. |
| Agent 3 (Pseudocode Writer) | Transcribe what the code does faithfully. If Agent 1's extraction report carries a `SUSPECTED_CODE_ISSUE` flag, include a prominent note in the pseudocode: "**Note:** This step may be incorrect; see discrepancy report REJ-...-HUMAN-NNN." |
| Agent 4 (Verification) | This is the primary checkpoint. When tracing a concrete example through both pseudocode and source code, if the output disagrees with the mathematical theorem, issue `SOURCE_CODE_BUG` per Section 14.2.7. Do not approve documentation that claims the code is correct when evidence suggests otherwise. |
| Agent 5 (Notation Compliance) | Not responsible for code correctness, but if a discrepancy note is missing from an artifact that Agent 1 flagged as `SUSPECTED_CODE_ISSUE`, report a `CROSS_REFERENCE_ERROR`. |

**What agents must never do:**

- Never modify source code, tests, build files, or any file outside `docs/mathref/` and `docs/notation_standard.md`
- Never construct a "proof" that is actually a rationalization of buggy behavior
- Never silently drop a theorem or weaken its statement to avoid having to report a bug
- Never assume that passing tests prove correctness (tests may be incomplete or wrong themselves)
- Never ignore a discrepancy between the mathematical derivation and the code on the grounds that "the code works in practice"

**Pipeline behavior when a code issue is found:**

1. The discovering agent issues `SOURCE_CODE_BUG` to the human (Section 14.2.7)
2. The pipeline **halts for the affected artifact** — no further agents process it
3. Other artifacts that do not depend on the suspect code may continue
4. The human resolves: fix the code, accept the behavior with a documented note, or determine the agent was wrong
5. After resolution, the pipeline restarts from Agent 1 for the affected artifact

### 14.1 Agent Roles

#### Agent 1: Codebase Analyst

**Purpose:** Extract the raw mathematical and algorithmic content from source code.

**Input:** Source files (C, C++, Python), header files, test files.

**Output:** Per-algorithm extraction report containing:
- The algorithm's purpose and the mathematical problem it solves
- All variables, their types, and their mathematical meaning
- The sequence of operations, translated from code to mathematical operations
- Invariants preserved at each step (extracted from assertions, comments, and loop structure)
- Preconditions and postconditions
- Numerical stability measures present in the code (e.g., Kahan summation, log-space computation)
- Complexity (time and space), derived from the code's loop structure
- **Suspected code issues** (if any): each flagged with `SUSPECTED_CODE_ISSUE`, citing the specific file, function, and line numbers, with a description of why the code appears inconsistent or incorrect

**Key source files this agent must be able to read:**

| Area | Files |
|------|-------|
| Core graph algorithms | `src/c/phasic.c`, `src/c/phasic.h`, `api/c/phasic.h` |
| Symbolic elimination | `src/c/phasic_symbolic.c` |
| Graph hashing | `src/c/phasic_hash.c` |
| Trace system | `src/c/trace/trace_internal.h`, `src/c/trace/trace_cache.c` |
| C++ bridge & GraphBuilder | `src/cpp/phasic_pybind.cpp`, `src/cpp/parameterized/graph_builder.cpp` |
| Trace elimination (Python) | `src/phasic/trace_elimination.py` |
| SVGD inference | `src/phasic/svgd.py` |
| MCMC inference | `src/phasic/mcmc.py` |
| BFFG importance weighting | `src/phasic/bffg.py` |
| Method of moments | `src/phasic/method_of_moments.py` |
| State indexing | `src/phasic/state_indexing.py` |
| HexGrid spatial models | `src/phasic/hex_grid.py` |

**Rules for this agent:**
- Must read `docs/notation_standard.md` before starting. Use the standard's symbol names when labeling extracted variables (e.g., label a sub-intensity matrix as **S**, not whatever the C variable is named).
- Extract what the code does, not what comments claim it does. Comments are hints, not truth.
- When code and comments disagree, report the discrepancy.
- Do not speculate about mathematical properties not evident in the code. Flag unknowns for the Proof Writer.

#### Agent 2: Mathematical Formalization Writer

**Purpose:** Transform extraction reports into rigorous mathematical definitions, theorems, and proofs following this notation standard.

**Input:** Extraction reports from Agent 1, `docs/notation_standard.md`, the three source papers (Røikjer et al. 2022, Hobolth et al. 2024, Bladt et al. 2011).

**Output:** Draft documentation sections in narrative definition-by-definition format (Section 10.1) containing:
- Formal definitions with all symbols drawn from Section 12 of this standard
- Theorems stating the mathematical properties each algorithm relies on
- Proofs of correctness for each algorithm
- Proofs of exactness for computed quantities (e.g., that the graph-based moment computation yields the same result as the matrix formula $\mathbb{E}[\tau^n] = n! \, \boldsymbol{\alpha}(-\mathbf{S})^{-n}\mathbf{e}$)
- Complexity analysis stated as formal propositions

**Rules for this agent:**
- Must read `docs/notation_standard.md` before writing any definition or equation. This is the authoritative source for all symbols.
- Every symbol used must appear in the notation standard. If a new symbol is needed, flag it — do not invent notation silently.
- Proofs must be self-contained: a reader should not need to look up external references to follow the argument, though references should be cited for attribution.
- When formalizing, preserve the correspondence to the code: include remarks like "In the implementation, this corresponds to the loop at line N of Algorithm M."
- Follow Section 8 (pseudocode conventions) exactly for all algorithm listings.

#### Agent 3: Algorithm Pseudocode Writer

**Purpose:** Produce clean, correct, and complete pseudocode for every algorithm, following Section 8 of this notation standard.

**Input:** Extraction reports from Agent 1, draft mathematical text from Agent 2, the actual source code.

**Output:** Numbered algorithm blocks (Algorithm 1, Algorithm 2, ...) with:
- Preamble stating preconditions
- Line-numbered pseudocode following Section 8.1 formatting rules
- Inline comments ($\triangleright$) explaining non-obvious steps
- Post-algorithm complexity statement
- Correspondence table mapping pseudocode variables to notation standard symbols and to source code variable names

**Rules for this agent:**
- Must read `docs/notation_standard.md` before writing any pseudocode. Follow Section 8 (pseudocode conventions) exactly. Use variable names from Section 12 (Symbol Quick Reference).
- The pseudocode must be faithful to what the code actually computes, not an idealized version.
- If the code handles edge cases (e.g., zero rewards, self-loops, absorbing vertices), the pseudocode must handle them too.
- Use the variable naming from the notation standard (Section 12), not from the source code.
- Include a correspondence table after each algorithm mapping pseudocode names ↔ code variable names ↔ mathematical symbols.

#### Agent 4: Verification Agent

**Purpose:** Independently verify every mathematical claim, proof, and algorithm in the documentation against the codebase and against the test suite.

**Input:** Complete draft documentation from Agents 2 and 3, the test suite, the source code.

**Output:** Verification report containing:
- For each theorem/proof: confirmation that the proof is logically valid, or a specific error with line reference
- For each algorithm: confirmation that the pseudocode matches the source code behavior, or a specific discrepancy
- For each claimed complexity: confirmation or correction
- For each equation: dimensional analysis (matrix dimensions, vector lengths agree)
- Cross-check against test suite: for each mathematical property claimed, cite the test(s) that exercise it, or flag the absence of a test

**Verification procedures:**

1. **Proof verification.** Read each proof line by line. For each logical step, confirm the inference is valid. Check:
   - Are quantifiers correct (∀ vs ∃)?
   - Does the induction base case actually hold?
   - Does the inductive step actually use the inductive hypothesis?
   - Are matrix dimensions consistent throughout?

2. **Algorithm-code correspondence.** For each algorithm, trace a concrete small example through both the pseudocode and the actual source code. Confirm they produce the same intermediate values. Use test fixtures from the test suite as examples.

3. **Equation verification.** For key equations, substitute a small concrete example (e.g., a 2-state or 3-state phase-type distribution) and verify numerically that both sides agree. Use values from existing tests where available.

4. **Notation compliance.** Confirm every symbol matches `notation_standard.md`. Flag any deviations.

**Key test files this agent must cross-reference:**

| Property | Test file |
|----------|-----------|
| Moment computation accuracy | `tests/pytest/test_graphbuilder_1d_correctness.py` |
| PMF normalization | `tests/pytest/test_graphbuilder_1d_correctness.py` |
| BFFG importance weight unbiasedness | `tests/pytest/test_bffg_accuracy.py` |
| Trace elimination equivalence | `tests/pytest/test_manual_vs_trace_graph.py` |
| SCC trace stitching | `tests/pytest/test_trace_stitching.py` |
| Multivariate correctness | `tests/pytest/test_multivariate_correctness.py` |
| SVGD posterior recovery | `tests/pytest/test_svgd_correctness.py` |
| MCMC convergence | `tests/pytest/test_mcmc_accuracy.py` |
| Sparse observations | `tests/pytest/test_sparse_observations.py` |

**Rules for this agent:**
- Must read `docs/notation_standard.md` before starting. Use it to verify dimensional consistency (e.g., that a claimed $p \times p$ matrix is actually used in contexts requiring that shape).
- This agent must not have seen the proofs during their construction. It acts as an independent reviewer.
- If a proof is wrong, do not fix it. Report the error to Agent 2 for correction.
- If an algorithm diverges from the code, report it — do not silently reconcile.

#### Agent 5: Notation Compliance and Consistency Checker

**Purpose:** Ensure that all documentation produced by Agents 2–4 is internally consistent and compliant with this notation standard.

**Input:** All draft documentation, `docs/notation_standard.md`.

**Output:** Compliance report containing:
- Every symbol used in the documentation, mapped to its definition in the notation standard
- Any symbol used but not defined in the standard (violation)
- Any symbol defined in the standard but used with a different meaning (violation)
- Any inconsistency between documents (e.g., Agent 2's definition of a quantity differs from Agent 3's pseudocode)
- Cross-reference integrity: every "see Definition X.Y" or "by Equation (Z)" actually points to something that exists
- Symbol index completeness: every symbol introduced on a page appears in that page's symbol index

**Rules for this agent:**
- This is a mechanical check, not a mathematical one. Do not evaluate proof correctness — that is Agent 4's job.
- Flag every deviation, no matter how minor. A misplaced bold or a missing subscript is a real error in a notation standard.

### 14.2 Workflow

The agents operate in a pipeline with feedback loops. The feedback loop mechanism, rejection report format, routing rules, re-verification requirements, termination conditions, and state tracking are specified in Sections 14.2.2 through 14.2.9.

#### 14.2.1 Pipeline Overview

```
┌─────────────────┐
│  Agent 1:       │
│  Codebase       │──── extraction reports ────┐
│  Analyst        │                            │
└────────▲────────┘                            ▼
         │                          ┌─────────────────────┐
         │ EXTRACTION_ERROR         │  Agent 2:           │
         │ EXTRACTION_AMBIGUITY     │  Math Formalization │──┐
         │ (from Agents 2,3,4)      │  Writer             │  │
         │                          └──────────▲──────────┘  │
         │                                     │             │
         │                          ┌──────────┘    draft    │
         │                          │  PROOF_LOGIC_ERROR     │
         │                          │  DEFINITION_ERROR      │
         │                          │  THEOREM_STATEMENT_ERR │
         │                          │  (from Agent 4)        │
         │                          ▼               text     │
         │                          ┌─────────────────────┐  │
         │                          │  Agent 3:           │  │
         │                          │  Algorithm          │  │
         │                          │  Pseudocode Writer  │  │
         │                          └──────────▲──────────┘  │
         │                                     │             │
         │                          ┌──────────┘             │
         │                          │  PSEUDOCODE_DIVERGENCE │
         │                          │  PSEUDOCODE_INCOMPLETE │
         │                          │  (from Agent 4)        │
         │                          ▼                        │
         │                ┌───────────────────┐              │
         │                │  Agent 4:         │── errors ────┘
         │                │  Verification     │
         └────────────────│  Agent            │
           SOURCE_CODE_BUG└───────────────────┘
           → human                  │
                                    │  verified draft
                                    ▼
                          ┌───────────────────┐
                          │  Agent 5:         │
                          │  Notation         │── NOTATION_VIOLATION
                          │  Compliance       │   → introducing agent
                          └───────────────────┘
                                    │
                                    │  compliant draft
                                    ▼
                          ┌───────────────────┐
                          │  Final review     │
                          │  (human)          │
                          └───────────────────┘
```

**Execution rules:**

1. **Agent 1 runs first** on the target source files. Its output is the input to Agents 2 and 3.
2. **Agents 2 and 3 may run in parallel** since they produce different artifacts (math text vs. pseudocode), but Agent 3 should incorporate Agent 2's definitions for variable naming.
3. **Agent 4 runs only after Agents 2 and 3 are complete.** It must receive the full draft, not incremental pieces.
4. **Agent 4's error reports route back** to the responsible agent (see Section 14.2.5 for routing rules). This loop repeats until Agent 4 produces a clean report.
5. **Agent 5 runs last**, after Agent 4 has signed off. Notation violations route back to the agent that introduced the symbol (tracked by provenance, see Section 14.2.9).
6. **The human reviews** the final compliant, verified draft. The human may request changes that restart from any agent.
7. **Any correction triggers re-verification** of all downstream agents that already processed the corrected artifact (see Section 14.2.6).

#### 14.2.2 Rejection Report Format

Every rejection from any agent must use the following structured format. The purpose is to give the receiving agent enough context to fix the error without re-reading the entire document.

**Required fields:**

| Field | Type | Description |
|-------|------|-------------|
| `report_id` | String | Unique identifier: `REJ-<rejecting_agent>-<receiving_agent>-<sequence>`, e.g., `REJ-A4-A2-003` |
| `rejecting_agent` | Integer (1–5) | The agent issuing the rejection |
| `receiving_agent` | Integer (1–5) or `human` | The agent responsible for fixing the error |
| `artifact` | String | The output artifact containing the error (e.g., "Definition 4.2", "Algorithm 3, line 7", "Theorem 5.1 proof, step 4") |
| `location` | Object | `{file, section, element_type, element_id}` — precise enough to find the error without searching |
| `criterion_violated` | String | The specific acceptance criterion from Section 14.4 (G1–G5) or verification procedure that failed |
| `severity` | Enum | `BLOCKING` or `WARNING` (see Section 14.2.3) |
| `error_class` | Enum | One of the error classes in Section 14.2.4 |
| `description` | String | What is wrong — specific enough that the receiving agent can locate and understand the error |
| `evidence` | String | Concrete evidence: the incorrect equation, the contradicting code snippet, the dimensional mismatch, the failing test case |
| `suggested_scope` | String | What the rejecting agent believes needs to change (advisory, not binding) |
| `context_snippet` | String | Verbatim excerpt (5–10 lines) of the artifact surrounding the error |

**Example:**

```
report_id: REJ-A4-A2-007
rejecting_agent: 4
receiving_agent: 2
artifact: Theorem 5.3 proof, step 4
location: {file: "algorithms/elimination.md", section: "5.3", element_type: "proof_step", element_id: 4}
criterion_violated: "Verification procedure 1: Does the inductive step use the inductive hypothesis?"
severity: BLOCKING
error_class: PROOF_LOGIC_ERROR
description: >
  The inductive step substitutes the Green matrix entry U_{ij} from the
  original graph, but the inductive hypothesis applies to G^{(i-1)}, not G.
  The step does not invoke the inductive hypothesis.
evidence: >
  Step 4 states "By the definition of U, we have..."
  Should state "By the inductive hypothesis applied to G^{(i-1)}, we have..."
suggested_scope: Rewrite step 4 to apply the inductive hypothesis to G^{(i-1)} explicitly.
context_snippet: |
  Step 3: ...hence G^{(i-1)} has the same absorption probabilities as G.
  Step 4: By the definition of U, we have U_{ij} = ...  ← ERROR
  Step 5: Substituting into the edge weight formula...
```

#### 14.2.3 Severity Levels

| Severity | Meaning | Pipeline effect |
|----------|---------|-----------------|
| `BLOCKING` | The artifact is incorrect in a way that invalidates downstream work | Pipeline halts for this artifact. Must be fixed before any downstream agent re-processes it. |
| `WARNING` | Deficiency that does not invalidate correctness (e.g., loose complexity bound, unnecessarily indirect proof, confusing but technically compliant notation) | Pipeline continues. Must be resolved before the artifact passes its final quality gate (Section 14.4). |

An artifact with any unresolved `BLOCKING` rejection cannot pass any quality gate. An artifact with only `WARNING` rejections may proceed but cannot reach `complete` status.

#### 14.2.4 Error Classification

Each rejection carries exactly one error class, which determines routing (Section 14.2.5).

| Error Class | Typical origin | Description |
|-------------|---------------|-------------|
| `EXTRACTION_ERROR` | Agent 1 | The extraction report misrepresents what the code does |
| `EXTRACTION_AMBIGUITY` | Agent 1 | Extraction is not wrong but too ambiguous for downstream agents |
| `PROOF_LOGIC_ERROR` | Agent 2 | A proof step is logically invalid |
| `DEFINITION_ERROR` | Agent 2 | A definition is incorrect, incomplete, or inconsistent with code |
| `THEOREM_STATEMENT_ERROR` | Agent 2 | A theorem does not match what the code computes |
| `PSEUDOCODE_DIVERGENCE` | Agent 3 | Pseudocode does not match source code behavior |
| `PSEUDOCODE_INCOMPLETE` | Agent 3 | Pseudocode omits an edge case present in source |
| `DIMENSION_MISMATCH` | Agent 2 or 3 | Matrix/vector dimensions inconsistent |
| `COMPLEXITY_ERROR` | Agent 2 or 3 | A claimed complexity bound is wrong |
| `NOTATION_VIOLATION` | Any agent | Symbol not in standard, or used with wrong meaning |
| `CROSS_REFERENCE_ERROR` | Any agent | Reference to nonexistent or wrong definition/equation/algorithm |
| `CONSISTENCY_ERROR` | Multiple agents | Two agents produced conflicting claims about the same quantity |
| `SOURCE_CODE_BUG` | None (special) | The source code itself appears to contain a bug (see Section 14.2.7) |

#### 14.2.5 Rejection Routing

All permitted rejection routes. Any route not listed here must be escalated to the human.

| Rejecting agent | Error class | Receiving agent |
|-----------------|-------------|-----------------|
| Agent 4 | `PROOF_LOGIC_ERROR` | Agent 2 |
| Agent 4 | `DEFINITION_ERROR` | Agent 2 |
| Agent 4 | `THEOREM_STATEMENT_ERROR` | Agent 2 |
| Agent 4 | `DIMENSION_MISMATCH` | Agent 2 or 3 (whichever artifact contains the mismatch; both if both) |
| Agent 4 | `COMPLEXITY_ERROR` | Agent 2 or 3 (whichever stated the complexity) |
| Agent 4 | `PSEUDOCODE_DIVERGENCE` | Agent 3 |
| Agent 4 | `PSEUDOCODE_INCOMPLETE` | Agent 3 |
| Agent 4 | `EXTRACTION_ERROR` | Agent 1 |
| Agent 4 | `EXTRACTION_AMBIGUITY` | Agent 1 |
| Agent 4 | `SOURCE_CODE_BUG` | Human (see Section 14.2.7) |
| Agent 5 | `NOTATION_VIOLATION` | The agent that introduced the symbol (per provenance tracking, Section 14.2.9) |
| Agent 5 | `CROSS_REFERENCE_ERROR` | The agent that wrote the reference |
| Agent 5 | `CONSISTENCY_ERROR` | Both conflicting agents; Agent 2's definitions take precedence, Agent 3 conforms |
| Agent 3 | `DEFINITION_ERROR` | Agent 2 (definition unusable for pseudocode) |
| Agent 3 | `EXTRACTION_AMBIGUITY` | Agent 1 |
| Agent 2 | `EXTRACTION_ERROR` | Agent 1 |
| Agent 2 | `EXTRACTION_AMBIGUITY` | Agent 1 |

**Upstream rejection rule:** Agents 2, 3, and 4 may reject to Agent 1, but only for `EXTRACTION_ERROR` and `EXTRACTION_AMBIGUITY`.

**No self-rejection:** An agent never sends a rejection to itself. Self-discovered errors are fixed directly and logged in the verification ledger (Section 14.2.9).

#### 14.2.6 Re-verification After Correction

**Principle:** Any downstream agent that already processed a corrected artifact must re-process it, because the correction may have invalidated its prior approval.

**Correction by Agent 1** (extraction report changed):
- Agents 2 and 3 re-run on the affected portion
- Then Agent 4 re-verifies
- Then Agent 5 re-checks
- Full downstream pipeline re-runs for the affected artifact

**Correction by Agent 2** (math text changed):
- Agent 3 re-checks pseudocode correspondence; updates if needed
- Agent 4 re-verifies corrected math text AND any revised pseudocode
- Agent 5 re-checks all revised artifacts

**Correction by Agent 3** (pseudocode changed):
- Agent 4 re-verifies the corrected pseudocode
- Agent 5 re-checks the corrected pseudocode
- Agent 2 does NOT re-run unless Agent 3's correction reveals Agent 2's definitions are also wrong (Agent 3 issues a rejection to Agent 2)

**Correction for notation fix** (in response to Agent 5):
- Only Agent 5 re-checks
- Agent 4 does NOT re-verify — unless the fix changed a symbol's meaning (not just typographic form), in which case Agent 5 flags this and Agent 4 re-verifies

**Scope:** Re-verification applies to the specific corrected artifact, not the entire document.

**Cascade:** If a correction causes a downstream agent to change its own artifact, and that change triggers a new rejection, it is a separate rejection cycle with its own round-trip counter.

#### 14.2.7 Source Code Bugs and Out-of-Scope Findings

**Source code bug.** If Agent 4 believes the source code is incorrect:

1. Agent 4 issues a rejection with `error_class: SOURCE_CODE_BUG`, `receiving_agent: human`.
2. The report must include: code location (file, function, line), expected behavior (from the math), actual behavior (from code tracing), and a concrete input demonstrating the discrepancy.
3. The pipeline **halts** for any artifact depending on the buggy code.
4. The human decides: (a) fix the code and restart from Agent 1, (b) document the code's actual behavior with a prominent divergence note, or (c) declare the discrepancy acceptable.
5. Agents must not silently work around a suspected code bug.

**Missing tests.** Reported as `WARNING` to the human. The documentation is not incorrect; it lacks test coverage.

**Ambiguous code.** Agent 1 must report explicitly. Agent 2 must state the assumption under which the formalization holds.

#### 14.2.8 Termination and Escalation

**Round-trip limits** (one round trip = rejection → correction → re-verification for one artifact between one pair of agents):

| Route | Max round trips |
|-------|----------------|
| Agent 4 → Agent 2 | 3 |
| Agent 4 → Agent 3 | 3 |
| Agent 4 → Agent 1 | 2 |
| Agent 5 → any agent | 2 |
| Agent 2 → Agent 1 | 2 |
| Agent 3 → Agent 2 | 2 |
| Agent 3 → Agent 1 | 2 |

**Escalation.** When the limit is reached:

1. The rejecting agent produces a **dispute report**: full rejection/correction history, summary of why the error persists, assessment of root cause.
2. Sent to the human. Pipeline halts for the affected artifact.
3. The human resolves by: providing a correction, siding with one agent, or modifying acceptance criteria.

**Agent disagreement protocol** (e.g., Agent 2 says proof is correct, Agent 4 says it is wrong):

1. Agent 2 responds with a **rebuttal** addressing Agent 4's specific evidence.
2. Agent 4 reviews. If it withdraws, the cycle ends. If it maintains the rejection, it produces a **refined rejection** addressing Agent 2's rebuttal point by point.
3. If unresolved after the max round trips, escalation occurs.
4. **No bare re-assertions.** Each round must introduce new evidence or argument. A response that does not substantively address the other agent's latest argument is treated as a concession.

#### 14.2.9 State Tracking: Verification Ledger

The pipeline maintains a **verification ledger** — one row per artifact (each definition, theorem, proof, algorithm, equation, or correspondence table).

**Columns:**

| Column | Description |
|--------|-------------|
| `artifact_id` | Unique ID, e.g., `def-4.2`, `thm-5.3`, `alg-3` |
| `artifact_type` | `definition`, `theorem`, `proof`, `algorithm`, `equation`, `correspondence_table` |
| `authoring_agent` | Agent 1, 2, or 3 |
| `current_status` | See status table below |
| `agent_N_status` | Per-agent status: `not_applicable`, `passed`, `rejected`, `pending_correction`, `pending_re_verification` |
| `open_rejections` | List of unresolved `BLOCKING` report IDs |
| `open_warnings` | List of unresolved `WARNING` report IDs |
| `round_trip_counts` | Map from `(rejecting, receiving)` to count |
| `disposition` | `in_pipeline`, `completed`, `escalated_to_human`, `blocked_by_code_bug` |

**Artifact statuses:**

| Status | Meaning |
|--------|---------|
| `draft` | Produced by authoring agent, not yet verified |
| `in_verification` | Agent 4 or 5 is processing |
| `rejected` | At least one `BLOCKING` rejection open |
| `pending_correction` | Receiving agent notified, correction not yet submitted |
| `correction_submitted` | Correction received, re-verification not started |
| `pending_re_verification` | Waiting for downstream re-verification per Section 14.2.6 |
| `verified` | Agent 4 approved (zero `BLOCKING` from Agent 4) |
| `compliant` | Agent 5 approved (zero notation violations) |
| `complete` | Both `verified` and `compliant`; all warnings resolved |
| `escalated` | Round-trip limit reached; waiting for human |
| `blocked` | Depends on another artifact that is `rejected`, `escalated`, or `blocked` |

**Provenance tracking.** For every symbol, the ledger records which agent introduced it:
- From Agent 1's extraction: provenance = Agent 1
- Introduced by Agent 2 in definitions: provenance = Agent 2
- Introduced by Agent 3 in pseudocode: provenance = Agent 3
- From notation standard directly: provenance = standard (misuse is charged to the using agent)

**Cascade tracking.** When correcting artifact X forces re-verification of artifact Y, the ledger sets Y to `pending_re_verification` with dependency note. Y cannot reach `complete` while X's correction is still propagating.

**New-error-from-correction scenario.** When Agent 2 corrects a proof and the correction introduces a new notation violation: Agent 4 re-verifies and approves the math → Agent 5 re-checks and finds the violation → Agent 5 issues a new rejection → the artifact reverts to `rejected` → Agent 2 fixes notation → Agent 5 re-checks (Agent 4 does not re-verify for notation-only changes).

### 14.3 What Each Agent Must Load Before Starting

**Every agent — without exception — must read the following files before producing any output.** This is not optional. An agent that produces a symbol not defined in the notation standard, or formats pseudocode inconsistently with Section 8, has failed.

| File | Purpose |
|------|---------|
| `docs/notation_standard.md` | Authoritative notation — all symbols, conventions, formatting rules, pseudocode conventions |
| `docs/mathref/PRODUCTION_PLAN.md` | SOM file structure, batch ordering, cross-reference rules, deduplication ownership |
| `docs/mathref/_template.md` | Within-file section template — mandatory structure for every SOM file |
| `CLAUDE.md` | Project context, development rules, API patterns |

Additionally:

| Agent | Additional required reading |
|-------|---------------------------|
| Agent 1 | The specific source files for the algorithm being documented |
| Agent 2 | The three source papers (Røikjer 2022, Hobolth 2024, Bladt 2011); Agent 1's extraction report |
| Agent 3 | Agent 1's extraction report; Agent 2's draft definitions |
| Agent 4 | The complete draft; the relevant test files; the relevant source files |
| Agent 5 | The complete draft only (plus `notation_standard.md`) |

### 14.4 Quality Gates

No documentation page is considered complete until all five gates pass:

| Gate | Agent | Criterion |
|------|-------|-----------|
| G1: Extraction complete | Agent 1 | Every function in the target source file has a corresponding extraction entry |
| G2: Formalization complete | Agent 2 | Every extracted algorithm has definitions, theorems, and proofs; all in narrative format |
| G3: Pseudocode complete | Agent 3 | Every algorithm has numbered pseudocode with correspondence table |
| G4: Verified | Agent 4 | Zero unresolved errors in the verification report |
| G5: Notation compliant | Agent 5 | Zero notation violations in the compliance report |

## 15. SOM Production Plan

The Supplementary Online Material (SOM) lives in `docs/mathref/`. Its file structure, production batches, cross-reference system, deduplication strategy, and final homogeneity pass are defined in:

> **`docs/mathref/PRODUCTION_PLAN.md`**

That document is the authoritative plan for SOM production. This notation standard governs the notation and agent behavior; the production plan governs what files exist, in what order they are produced, and how they reference each other.

All agents producing SOM content must read both this standard and the production plan before starting.

## 16. Invoking Agents: Practical Operations Guide

This section provides copy-paste prompts for the two most common operations: adding documentation for new features, and proofreading edits.

### 16.1 Adding Documentation for a New Feature or Algorithm

When new code is added to phasic and needs formal documentation in the SOM, use this prompt to invoke the full 5-agent pipeline:

> **Prompt to Claude:**
>
> I have added a new feature/algorithm to phasic. Please document it in the SOM using the 5-agent pipeline defined in `docs/notation_standard.md` Section 14.
>
> **What changed:** [describe the new feature, which source files were added/modified]
>
> **Target SOM file:** [either an existing file number to extend, or "new file NN_name.md"]
>
> Please:
> 1. Read `docs/notation_standard.md`, `docs/mathref/PRODUCTION_PLAN.md`, and `docs/mathref/_template.md` first
> 2. Run Agent 1 (Codebase Analyst) on the source files I mentioned
> 3. Run Agent 2 (Math Formalization) to produce definitions, theorems, and proofs
> 4. Run Agent 3 (Algorithm Pseudocode) to produce pseudocode with correspondence tables
> 5. Run Agent 4 (Verification) to verify proofs and algorithm-code correspondence
> 6. Run Agent 5 (Notation Compliance) to check notation standard compliance
> 7. Update `docs/mathref/00_index.md` registries with any new algorithms or definitions

**If the new feature fits into an existing SOM file** (e.g., a new sampling method goes into `17_sampling.md`):
- Agents 2 and 3 append to the existing file, continuing the definition/theorem numbering
- Agent 4 verifies both the new content and that it is consistent with the existing content
- Agent 5 checks the full file

**If the new feature requires a new SOM file** (e.g., a completely new inference method):
- The new file gets the next available number (28, 29, ...)
- Update `PRODUCTION_PLAN.md` Section 2.2 (file listing) and Section 6.1 (deduplication ownership)
- Update `00_index.md` file listing, dependency DAG, and algorithm registry
- The new file must follow `_template.md` exactly

### 16.2 Proofreading Edits for Math, Logic, and Consistency

When you have manually edited SOM files (for clarity, to fix exposition, to add examples, etc.) and want to verify your edits haven't broken anything, use this prompt:

> **Prompt to Claude:**
>
> I have edited the following SOM files: [list files]. Please proofread my edits using Agents 4 and 5 from `docs/notation_standard.md` Section 14.
>
> Please:
> 1. Read `docs/notation_standard.md` (Sections 2-8, 11, 12) and the edited files
> 2. Run Agent 4 (Verification):
>    - Verify all proofs in the edited files are still logically valid
>    - Check dimensional consistency of all equations
>    - Trace one concrete example through any modified algorithm
>    - Verify cross-references to other SOM files still point to existing definitions
> 3. Run Agent 5 (Notation Compliance):
>    - Check all symbols against the notation standard
>    - Verify no new forward references were introduced
>    - Verify the symbol index is still complete
>    - Check template section order
> 4. If my edits changed any definitions or theorems that are referenced by OTHER files, check those downstream files too for consistency
>
> Report any issues as PASS/FAIL with exact locations.

**Key: the "downstream check."** If you edit a definition in file 06 that is referenced by files 07, 10, 11, and 13, those files need re-checking. The prompt above asks for this explicitly. Claude should:

1. Grep for references to the edited definition across all SOM files
2. Verify each reference still makes sense after the edit
3. If an Extended Recall in another file quotes the edited definition verbatim, verify the quote still matches

### 16.3 Quick Notation Check (Single File)

For a fast check of a single file without full proof verification:

> **Prompt to Claude:**
>
> Please run Agent 5 (Notation Compliance) from `docs/notation_standard.md` Section 14 on `docs/mathref/NN_filename.md`. Check symbols, template compliance, cross-references, numbering, and symbol index completeness.

### 16.4 Full Cross-SOM Consistency Check

To re-run the homogeneity pass from `PRODUCTION_PLAN.md` Section 8 after a series of edits:

> **Prompt to Claude:**
>
> Please run the full homogeneity pass (H1-H5) from `docs/mathref/PRODUCTION_PLAN.md` Section 8 across all SOM files. Check: H1 (internal consistency — cross-references, algorithm counter, definition uniqueness), H4 (notation compliance sweep), and flag any issues.

H2 (exhaustiveness) and H3 (non-duplication) are typically only needed after adding new files, not after editing existing ones.

### 16.5 What Each Agent Invocation Looks Like in Practice

When Claude runs the agent pipeline, it:

1. **Agent 1** — Spawns a sub-agent that reads the source files and produces an extraction report (not written to disk; passed to Agents 2/3)
2. **Agents 2+3** — Spawns one or two sub-agents that write the actual `.md` file content
3. **Agent 4** — Spawns a sub-agent that reads the written file and the source code, produces a PASS/FAIL verification report
4. **Agent 5** — Spawns a sub-agent that reads the written file and the notation standard, produces a compliance report
5. **Fixes** — Claude applies fixes based on Agent 4/5 reports, then re-runs the failing checks

For proofreading (Section 16.2), only steps 3-5 run, since the content already exists.

### 16.6 When NOT to Use the Full Pipeline

- **Fixing a typo**: Just fix it. No agents needed.
- **Updating an Implementation Notes section** (line numbers changed, function renamed): Just update it. Run Agent 5 (quick notation check) if you want.
- **Adding a Remark or Example**: Run Agent 4 on the file to verify the example is correct. Agent 5 is optional.
- **Changing notation standard itself**: Follow Section 13 (Updating This Standard) — that has its own procedure with impact analysis.

## 17. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-08 | Initial draft |
| 1.1 | 2026-04-08 | Added Section 13: procedure for updating notation, impact analysis, search-replace workflow, journal overrides, governance |
| 1.2 | 2026-04-08 | Added Section 14: five-agent team specification with workflow, quality gates, required reading, and source file mapping |
| 1.3 | 2026-04-08 | Expanded Section 14.2 with subsections 14.2.1–14.2.9: rejection report format, severity levels, error classification, routing table, re-verification rules, source code bug handling, termination/escalation with round-trip limits, dispute protocol, verification ledger with provenance and cascade tracking |
| 1.4 | 2026-04-08 | Added Section 14.0: "Code Correctness Is Not Assumed" foundational principle — per-agent obligations when code appears wrong, explicit prohibition on fabricating proofs or weakening theorems, SUSPECTED_CODE_ISSUE flag in extraction reports, never-modify-code rule |
| 1.5 | 2026-04-08 | Added Section 15: SOM production plan reference. Updated Section 14.3 required reading to include `PRODUCTION_PLAN.md` and `_template.md` |
| 1.6 | 2026-04-09 | Added Section 2.3: Named constants and operation types must use `\texttt{}` (monospace) in math mode, not `\textsc{}` (unsupported by MathJax). Added row to Section 2.1 typographic conventions table. |
| 1.7 | 2026-04-09 | Added Section 16: Practical operations guide — copy-paste prompts for adding new documentation, proofreading edits, quick notation checks, full consistency checks, and when NOT to use the pipeline. |
