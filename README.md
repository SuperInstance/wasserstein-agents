# wasserstein-agents

**Wasserstein distance, optimal transport, and agent distribution coordination.**

A Rust library for computing optimal transport plans, Wasserstein distances, and evolving agent distributions via gradient flow. Built on the Sinkhorn-Knopp algorithm with log-domain numerical stability.

## What This Does

This library provides three things:

1. **Sinkhorn solver** — Compute entropy-regularized optimal transport plans between discrete distributions.
2. **Wasserstein distances** — W₁ and W₂ metrics between probability distributions.
3. **JKO gradient flow** — Evolve distributions over time using the Jordan-Kinderlehrer-Otto scheme.

The "agents" framing: model a fleet of agents as a probability distribution over state space, then use optimal transport to coordinate, compare, and evolve them.

## Key Idea

The **Wasserstein distance** (Earth Mover's Distance) measures how much "work" it takes to reshape one distribution into another. Unlike KL divergence, it respects the geometry of the underlying space — two point masses that are close cost less to transport than far-apart ones.

The **Sinkhorn algorithm** computes an approximate optimal transport plan by adding entropy regularization. This makes the problem strictly convex and solvable via iterative row/column normalization of a Gibbs kernel:

```
K = exp(-C / ε)
```

where C is the cost matrix and ε controls regularization strength.

The **JKO scheme** discretizes Wasserstein gradient flow as a sequence of proximal steps:

```
μ_{t+1} = argmin_ν { τ·F(ν) + W₂²(μ_t, ν) / 2 }
```

This evolves a distribution toward the minimum of an energy functional F, in the geometry of optimal transport.

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
wasserstein-agents = "0.1.0"
```

Or use `cargo add`:

```sh
cargo add wasserstein-agents
```

Requires Rust 2021 edition. No external dependencies — pure Rust.

## Quick Start

```rust
use wasserstein_agents::{SinkhornSolver, OptimalTransport, AgentDistribution};

// Define a cost matrix
let cost = vec![
    vec![0.0, 1.0, 2.0],
    vec![1.0, 0.0, 1.0],
    vec![2.0, 1.0, 0.0],
];
let mu = vec![0.5, 0.3, 0.2];
let nu = vec![0.2, 0.3, 0.5];

// Compute Wasserstein-2 distance
let w2 = OptimalTransport::wasserstein_2(&cost, &mu, &nu);
println!("W₂ = {}", w2);

// Get the full transport plan
let solver = SinkhornSolver::new(0.1);
let plan = solver.solve(&cost, &mu, &nu);
// plan[i][j] = mass transported from source i to target j

// Work with agent distributions
let fleet = AgentDistribution::uniform(vec![
    vec![0.0, 0.0],
    vec![1.0, 0.0],
    vec![0.0, 1.0],
]);
let targets = AgentDistribution::uniform(vec![
    vec![1.0, 1.0],
    vec![2.0, 0.0],
    vec![0.0, 2.0],
]);

// Distance between two agent configurations
let dist = fleet.wasserstein_distance(&targets);

// Optimal assignment plan
let assignment = fleet.optimal_assignment(&targets);
```

### Gradient Flow

```rust
use wasserstein_agents::{AgentDistribution, JKOScheme};

let initial = AgentDistribution::uniform(vec![
    vec![3.0], vec![-3.0], vec![1.0], vec![-1.0],
]);

// Evolve toward the origin (heat equation / quadratic potential)
let jko = JKOScheme::new(0.1, 50);
let trajectory = jko.flow_to_origin(&initial);

// Each step moves the distribution closer to the minimum
for (t, dist) in trajectory.iter().enumerate() {
    println!("t={}: mean = {:?}", t, dist.mean());
}

// Custom potential: V(x) = 0.5|x|² → ∇V(x) = x
let grad_v = |x: &[f64]| x.to_vec();
let custom_traj = jko.flow_with_potential(&initial, grad_v);
```

## API Reference

### `SinkhornSolver`

Entropy-regularized optimal transport via the Sinkhorn-Knopp algorithm.

| Method | Description |
|--------|-------------|
| `new(regularization)` | Create solver with entropy parameter ε |
| `solve(cost, mu, nu) → Vec<Vec<f64>>` | Compute transport plan T[i][j] |
| `transport_cost(plan, cost) → f64` | Total cost of a transport plan |

**Parameters:**
- `regularization` — Entropy strength ε. Smaller → closer to exact OT, but slower convergence and potential numerical issues. Typical: 0.01–0.1.
- `max_iterations` — Defaults to 1000.
- `tolerance` — Convergence threshold. Defaults to 1e-8.

### `OptimalTransport`

Static methods for Wasserstein distances.

| Method | Description |
|--------|-------------|
| `wasserstein_1(cost, mu, nu) → f64` | W₁ distance (Earth Mover's) |
| `wasserstein_2(cost, mu, nu) → f64` | W₂ distance (square root) |
| `wasserstein_2_squared(cost, mu, nu) → f64` | W₂² distance |
| `barycenter(distributions, weights, n_iter) → Vec<f64>` | Fréchet mean of distributions |

### `AgentDistribution`

A probability distribution over agent positions in state space.

| Method | Description |
|--------|-------------|
| `uniform(positions)` | Uniform weights over given positions |
| `weighted(positions, weights)` | Explicit weights |
| `len() → usize` | Number of agents |
| `dimension() → usize` | State space dimension |
| `mean() → Vec<f64>` | Center of mass |
| `covariance() → Vec<Vec<f64>>` | Covariance matrix |
| `distance_matrix() → Vec<Vec<f64>>` | Pairwise Euclidean distances |
| `wasserstein_distance(other) → f64` | W₂ to another distribution |
| `optimal_assignment(targets) → Vec<Vec<f64>>` | Transport plan to targets |
| `spread(factor) → AgentDistribution` | Dilate positions from centroid |

### `JKOScheme`

Wasserstein gradient flow via the Jordan-Kinderlehrer-Otto scheme.

| Method | Description |
|--------|-------------|
| `new(dt, n_steps)` | Create scheme with time step and step count |
| `flow_to_origin(initial) → Vec<AgentDistribution>` | Quadratic potential flow |
| `flow_with_potential(initial, grad_v) → Vec<AgentDistribution>` | Custom potential flow |
| `wasserstein_trajectory(trajectory) → Vec<f64>` | W₂ distances between consecutive steps |

## How It Works

### Sinkhorn Algorithm

1. **Build the Gibbs kernel**: K = exp(-C/ε) where C is the cost matrix.
2. **Iterate in log domain**: Maintain dual variables u, v. Alternately project onto row constraints (u update) and column constraints (v update).
3. **Recover the plan**: T[i][j] = exp(u[i]) · K[i][j] · exp(v[j]).
4. **Convergence check**: Stop when dual variables change by less than the tolerance.

Log-domain stabilization prevents numerical underflow when ε is small.

### Wasserstein Distances

- **W₁**: Sinkhorn with small ε (≈0.01), then dot-product of plan with cost.
- **W₂²**: Same computation with squared-distance cost matrix.
- **W₂**: Square root of W₂².

### JKO Scheme

Each JKO step solves a regularized optimal transport problem:

1. Compute the proximal update: move positions toward the energy minimum.
2. Solve OT between current and candidate distributions.
3. Update weights from the transport plan's column marginals.
4. Normalize to maintain total mass = 1.

For the quadratic potential V(x) = ½|x|², the proximal operator is simply x → x/(1 + τ).

### Barycenter

The Wasserstein barycenter minimizes the weighted sum of Wasserstein distances to a set of input distributions. Computed via fixed-point iteration: alternately transport each input distribution toward the current barycenter estimate, then update the barycenter as the weighted average of the push-forwards.

## The Math

### Optimal Transport Problem

Given source distribution μ ∈ Δⁿ, target ν ∈ Δᵐ, and cost matrix C ∈ ℝⁿˣᵐ:

$$\min_{T \in \Pi(\mu, \nu)} \sum_{i,j} C_{ij} T_{ij}$$

where Π(μ, ν) is the set of transport plans with marginals μ (rows) and ν (columns).

### Entropy-Regularized OT (Sinkhorn)

$$\min_{T \in \Pi(\mu, \nu)} \sum_{i,j} C_{ij} T_{ij} + \varepsilon \sum_{i,j} T_{ij} \log T_{ij}$$

The entropy term makes the problem strictly convex. The optimal solution has the form:

$$T_{ij} = \exp(u_i) \cdot K_{ij} \cdot \exp(v_j)$$

where K = exp(-C/ε) and u, v are the Sinkhorn dual variables found by alternating projection.

### Wasserstein-p Distance

$$W_p(\mu, \nu) = \left( \min_{T \in \Pi(\mu, \nu)} \sum_{i,j} |x_i - x_j|^p \cdot T_{ij} \right)^{1/p}$$

For p = 1 this is the Earth Mover's Distance. For p = 2 it gives the natural Riemannian structure on the space of probability distributions (the Wasserstein-2 geometry).

### JKO Gradient Flow

The gradient flow of a functional F in Wasserstein-2 geometry is discretized as:

$$\mu^{n+1} = \arg\min_{\nu} \left\{ \tau \cdot F(\nu) + \frac{1}{2} W_2^2(\mu^n, \nu) \right\}$$

For F(ν) = ∫ ½|x|² dν (quadratic potential), the flow is the heat equation, and distributions converge to a Dirac delta at the origin.

### Wasserstein Barycenter

$$\bar{\mu} = \arg\min_{\nu} \sum_{k=1}^{K} \lambda_k \cdot W_2^2(\nu, \mu_k)$$

where λₖ are non-negative weights summing to 1. This generalizes the Euclidean mean to the curved Wasserstein space.

## Test Coverage

16 tests covering:
- Sinkhorn convergence on identity-cost and asymmetric distributions
- Marginal preservation verification
- W₁ and W₂ on identical and different distributions
- Transport cost computation
- Agent distribution statistics (mean, covariance, distance matrix)
- JKO convergence to origin, trajectory length, mass preservation
- Custom potential gradient flow
- Wasserstein trajectory between time steps

## License

MIT
