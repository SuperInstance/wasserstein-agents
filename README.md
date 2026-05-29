# wasserstein-agents

Wasserstein distance, optimal transport, and agent distribution coordination via the Sinkhorn algorithm and JKO gradient flow.

## Usage

```rust
use wasserstein_agents::{SinkhornSolver, OptimalTransport, AgentDistribution, JKOScheme};

// Sinkhorn optimal transport
let solver = SinkhornSolver::new(0.1);
let cost = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
let plan = solver.solve(&cost, &[0.5, 0.5], &[0.5, 0.5]);

// Agent distributions
let fleet = AgentDistribution::uniform(vec![vec![0.0], vec![2.0]]);
let mean = fleet.mean(); // [1.0]
let cov = fleet.covariance();

// JKO gradient flow toward origin
let jko = JKOScheme::new(0.1, 50);
let trajectory = jko.flow_to_origin(&fleet);
```

## Features

- **Sinkhorn algorithm**: Entropy-regularized optimal transport with log-domain stability
- **Wasserstein-1 and Wasserstein-2** distance computation
- **AgentDistribution**: Mean, covariance, distance matrix, spread, optimal assignment
- **JKO scheme**: Wasserstein gradient flow (Jordan-Kinderlehrer-Otto)
- **Distribution barycenter**: Fixed-point iteration for Wasserstein barycenters

## Tests

16 tests, all passing. `cargo test` to run.

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
