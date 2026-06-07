# wasserstein-agents

Measure how different your agents are. Make them converge. The Wasserstein distance is the cost of turning one distribution into another — literally the price of moving the dirt.

```rust
use wasserstein_agents::*;
```

---

## 1. Earth Mover's Distance: The Intuition

Agent A produces outputs clustered around [1, 2, 3]. Agent B produces outputs around [4, 5, 6]. How far apart are they? Not "are any values the same?" but "how much total work to morph A into B?"

```rust
use wasserstein_agents::AgentDistribution;

// Agent A: conservative, outputs near 1-3
let agent_a = AgentDistribution::uniform(vec![
    vec![1.0], vec![2.0], vec![3.0],
]);

// Agent B: aggressive, outputs near 4-6
let agent_b = AgentDistribution::uniform(vec![
    vec![4.0], vec![5.0], vec![6.0],
]);

// Earth Mover's Distance: how much work to turn A into B?
let distance = agent_a.wasserstein_distance(&agent_b);
println!("W₂(A, B) = {:.3}", distance);
// ~5.0 — you'd need to shift each point by ~3 units on average
// This IS a metric: symmetric, non-negative, triangle inequality

// Compare with itself — should be ~0
let self_dist = agent_a.wasserstein_distance(&agent_a);
println!("W₂(A, A) = {:.3} (should be ~0)", self_dist);
```

---

## 2. Agent Distributions: Mean, Covariance, Spread

```rust
use wasserstein_agents::AgentDistribution;

// An agent that produces 2D outputs: [latency, throughput]
let agent = AgentDistribution::weighted(
    vec![
        vec![12.0, 950.0],  // normal run
        vec![13.0, 940.0],  // normal run
        vec![11.0, 960.0],  // normal run
        vec![45.0, 500.0],  // degraded run
    ],
    vec![0.3, 0.3, 0.3, 0.1], // mostly normal, sometimes degraded
);

println!("{} samples, {}D state space", agent.len(), agent.dimension());

let mean = agent.mean();
println!("Mean: [{:.1}, {:.1}]", mean[0], mean[1]);
// Mean latency ~17.5, throughput ~870

let cov = agent.covariance();
println!("Covariance:");
println!("  Var(latency)     = {:.1}", cov[0][0]);
println!("  Var(throughput)  = {:.1}", cov[1][1]);
println!("  Cov(lat, tput)   = {:.1}", cov[0][1]);
// High variance in latency = inconsistent performance

// Distance matrix: how far apart are the agent's own samples?
let dm = agent.distance_matrix();
println!("Distance from sample 0 to 3: {:.1}", dm[0][3]);
// The degraded run (sample 3) is far from normal runs
```

### Spreading Agents Apart

```rust
let spread = agent.spread(2.0); // move each sample 2x away from centroid
let mean_before = agent.mean();
let mean_after = spread.mean();
println!("Mean before: [{:.1}, {:.1}]", mean_before[0], mean_before[1]);
println!("Mean after:  [{:.1}, {:.1}]", mean_after[0], mean_after[1]);
// Mean stays the same — spreading is symmetric around the centroid
let spread_dist = agent.wasserstein_distance(&spread);
println!("W₂(original, spread) = {:.1}", spread_dist);
```

---

## 3. Sinkhorn Algorithm: Step by Step

The Sinkhorn algorithm solves optimal transport by adding entropy regularization. It alternates between normalizing rows and columns.

```rust
use wasserstein_agents::SinkhornSolver;

// Cost matrix: cost[i][j] = cost of moving mass from position i to position j
let cost = vec![
    vec![0.0, 1.0, 2.0, 3.0],
    vec![1.0, 0.0, 1.0, 2.0],
    vec![2.0, 1.0, 0.0, 1.0],
    vec![3.0, 2.0, 1.0, 0.0],
];

let mu = vec![0.25, 0.25, 0.25, 0.25]; // uniform source
let nu = vec![0.25, 0.25, 0.25, 0.25]; // uniform target

let solver = SinkhornSolver::new(0.1); // regularization ε=0.1
let plan = solver.solve(&cost, &mu, &nu);

println!("Transport plan:");
for (i, row) in plan.iter().enumerate() {
    print!("  from {}: [", i);
    for (j, &val) in row.iter().enumerate() {
        print!("{:.3} ", val);
    }
    println!("]");
}
// When source = target (both uniform), the plan is approximately diagonal
// Mass stays put when there's no reason to move it

let total_cost = SinkhornSolver::transport_cost(&plan, &cost);
println!("Total transport cost: {:.4}", total_cost);
// Should be close to 0 — nothing needs to move
```

### Asymmetric Distributions

```rust
use wasserstein_agents::SinkhornSolver;

let cost = vec![
    vec![0.0, 1.0, 4.0],
    vec![1.0, 0.0, 1.0],
    vec![4.0, 1.0, 0.0],
];

let mu = vec![0.5, 0.5, 0.0]; // mass at positions 0 and 1
let nu = vec![0.0, 0.5, 0.5]; // mass at positions 1 and 2

let solver = SinkhornSolver::new(0.05);
let plan = solver.solve(&cost, &mu, &nu);

println!("Moving mass from [0.5, 0.5, 0] to [0, 0.5, 0.5]:");
for (i, row) in plan.iter().enumerate() {
    println!("  from {} (mass {:.1}): to [{:.3}, {:.3}, {:.3}]",
        i, mu[i], row[0], row[1], row[2]);
}

// Verify marginals
for (i, row) in plan.iter().enumerate() {
    let row_sum: f64 = row.iter().sum();
    println!("Row {} sum: {:.3} (should be {:.3})", i, row_sum, mu[i]);
}
for j in 0..3 {
    let col_sum: f64 = plan.iter().map(|r| r[j]).sum();
    println!("Col {} sum: {:.3} (should be {:.3})", j, col_sum, nu[j]);
}
```

---

## 4. Wasserstein-1 vs Wasserstein-2

```rust
use wasserstein_agents::OptimalTransport;

let cost = vec![
    vec![0.0, 1.0, 4.0, 9.0],
    vec![1.0, 0.0, 1.0, 4.0],
    vec![4.0, 1.0, 0.0, 1.0],
    vec![9.0, 4.0, 1.0, 0.0],
];

let mu = vec![0.5, 0.5, 0.0, 0.0];
let nu = vec![0.0, 0.0, 0.5, 0.5];

let w1 = OptimalTransport::wasserstein_1(&cost, &mu, &nu);
let w2 = OptimalTransport::wasserstein_2(&cost, &mu, &nu);

println!("W₁ = {:.3}", w1); // linear cost: Σ cᵢⱼ Tᵢⱼ
println!("W₂ = {:.3}", w2); // quadratic cost: √(Σ cᵢⱼ² Tᵢⱼ)
// W₂ penalizes long-distance moves more harshly
// W₁ treats all moves linearly — "1 unit of dirt moved 3 units costs 3"
// W₂ squres the distance — "1 unit of dirt moved 3 units costs 9"

let w2_sq = OptimalTransport::wasserstein_2_squared(&cost, &mu, &nu);
println!("W₂² = {:.3}", w2_sq);
```

---

## 5. Comparing Two Agent Fleets

```rust
use wasserstein_agents::AgentDistribution;

// Fleet A: 3 agents producing embeddings in 2D
let fleet_a = AgentDistribution::uniform(vec![
    vec![0.0, 1.0],
    vec![1.0, 0.0],
    vec![0.5, 0.5],
]);

// Fleet B: same agents after fine-tuning — embeddings shifted
let fleet_b = AgentDistribution::uniform(vec![
    vec![0.1, 1.2],
    vec![1.1, 0.1],
    vec![0.6, 0.4],
]);

let dist = fleet_a.wasserstein_distance(&fleet_b);
println!("W₂(fleet A, fleet B) = {:.3}", dist);
// Small distance = fine-tuning didn't break things

// Optimal assignment: which agent in A maps to which in B?
let plan = fleet_a.optimal_assignment(&fleet_b);
println!("Optimal transport plan:");
for (i, row) in plan.iter().enumerate() {
    let best_j = row.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(j, _)| j);
    println!("  A[{}] → B[{}] (mass {:.3})",
        i, best_j.unwrap(), row[best_j.unwrap()]);
}
```

---

## 6. JKO Gradient Flow: Agents Evolving Over Time

The Jordan-Kinderlehrer-Otto (JKO) scheme evolves a distribution by minimizing transport cost + functional cost at each step. It's gradient descent *in the space of distributions*.

```rust
use wasserstein_agents::{AgentDistribution, JKOScheme};

// Start with agents scattered far from origin
let initial = AgentDistribution::uniform(vec![
    vec![5.0], vec![-3.0], vec![2.0], vec![-4.0], vec![1.0],
]);

// JKO with quadratic potential: drives agents toward origin
let jko = JKOScheme::new(0.1, 50); // dt=0.1, 50 steps
let trajectory = jko.flow_to_origin(&initial);

println!("Gradient flow trajectory:");
for (step, dist) in trajectory.iter().enumerate() {
    let mean = dist.mean();
    if step % 10 == 0 || step == trajectory.len() - 1 {
        println!("  t={:5.1}: mean={:.3}, positions={:?}",
            step as f64 * 0.1,
            mean[0],
            dist.positions.iter().map(|p| format!("{:.2}", p[0]))
                .collect::<Vec<_>>());
    }
}

let final_dist = trajectory.last().unwrap();
let final_mean = final_dist.mean();
println!("\nFinal mean: [{:.4}] (converging to 0)", final_mean[0]);

// Track the Wasserstein distance between consecutive steps
let w_traj = jko.wasserstein_trajectory(&trajectory);
println!("Step sizes (W₂ between consecutive distributions):");
for (i, &w) in w_traj.iter().enumerate() {
    if i < 5 || i >= w_traj.len() - 2 {
        println!("  step {}: W₂ = {:.4}", i, w);
    }
}
// Steps get smaller as the distribution converges
```

### Custom Potential: Drive Agents to Any Target

```rust
use wasserstein_agents::{AgentDistribution, JKOScheme};

let initial = AgentDistribution::uniform(vec![
    vec![0.0, 0.0],
    vec![1.0, 1.0],
    vec![-1.0, 2.0],
]);

// Potential: V(x,y) = (x-3)² + (y-3)² → drives to (3,3)
let grad_v = |p: &[f64]| vec![2.0 * (p[0] - 3.0), 2.0 * (p[1] - 3.0)];

let jko = JKOScheme::new(0.05, 100);
let traj = jko.flow_with_potential(&initial, grad_v);

let final_pos = &traj.last().unwrap().positions;
println!("After 100 steps:");
for (i, p) in final_pos.iter().enumerate() {
    println!("  agent {}: ({:.2}, {:.2})", i, p[0], p[1]);
}
// All agents should be near (3, 3)
```

---

## 7. Barycenter: The "Average" Distribution

The Wasserstein barycenter is the Fréchet mean in distribution space — the distribution that minimizes total transport cost to all input distributions.

```rust
use wasserstein_agents::OptimalTransport;

// Three agent distributions (1D, on a shared grid of 5 points)
let grid = vec![0.0, 1.0, 2.0, 3.0, 4.0];

// Distribution 1: peaked at 0
let mu1 = vec![0.5, 0.3, 0.1, 0.05, 0.05];
// Distribution 2: peaked at 4
let mu2 = vec![0.05, 0.05, 0.1, 0.3, 0.5];
// Distribution 3: peaked at 2 (middle)
let mu3 = vec![0.1, 0.2, 0.4, 0.2, 0.1];

let cost: Vec<Vec<f64>> = (0..5)
    .map(|i| (0..5).map(|j| (grid[i] - grid[j]).powi(2)).collect())
    .collect();

let distributions = vec![
    (&mu1[..], &cost),
    (&mu2[..], &cost),
    (&mu3[..], &cost),
];
let weights = vec![1.0/3.0, 1.0/3.0, 1.0/3.0];

let bary = OptimalTransport::barycenter(&distributions, &weights, 20);

println!("Barycenter distribution:");
for (i, &v) in bary.iter().enumerate() {
    println!("  x={:.0}: {:.3} {}", grid[i], v, "█".repeat((v * 50.0) as usize));
}
// Should be peaked in the middle — the "consensus" of the three distributions
// Unlike Euclidean averaging of probabilities (which is just the arithmetic mean),
// the Wasserstein barycenter respects the geometry of the underlying space
```

---

## The Full Picture

```
Agent A (distribution)    Agent B (distribution)
        ↓                         ↓
        └─── cost matrix ────────┘
                     ↓
              Sinkhorn solver
                     ↓
           Transport plan T[i][j]
           "how much mass from i to j"
                     ↓
         W₁ = Σ cᵢⱼ Tᵢⱼ     (linear cost)
         W₂ = √(Σ cᵢⱼ² Tᵢⱼ)  (quadratic cost)
                     ↓
        ┌────────────┼────────────┐
        ↓            ↓            ↓
    Compare     Barycenter    Gradient flow
    agents      (average)     (converge)
```

- **W₁**: "Total fuel to move the dirt." Linear. Treats all distances equally.
- **W₂**: "Total fuel²." Quadratic. Penalizes long-range moves. Has a Riemannian structure — you can do gradient descent in distribution space.
- **Barycenter**: The "average" distribution. Respects geometry, unlike arithmetic mean.
- **JKO flow**: Gradient descent in Wasserstein space. Your agents converge to a target distribution over time.

---

## API Reference

| Type | What it does |
|------|-------------|
| `AgentDistribution` | Positions + weights. Mean, covariance, distance matrix. Wasserstein distance. |
| `SinkhornSolver` | Entropy-regularized optimal transport. Returns transport plan matrix. |
| `OptimalTransport` | W₁, W₂, W₂² distances. Barycenter computation. |
| `JKOScheme` | Wasserstein gradient flow. Quadratic potential, custom potentials, trajectory tracking. |
