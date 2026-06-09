//! Tutorial: Wasserstein Agents — Optimal Transport for Fleet Coordination
//!
//! Run with: cargo run --example tutorial

use wasserstein_agents::*;

fn main() {
    println!("=== Lesson 1: Agent Distributions ===\n");
    {
        // An AgentDistribution is a probability distribution over agent states
        let fleet = AgentDistribution::uniform(vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ]);
        println!("Fleet of {} agents at 2D positions:", fleet.len());
        for (i, (pos, w)) in fleet.positions.iter().zip(&fleet.weights).enumerate() {
            println!("  Agent {}: pos={:?}, weight={:.3}", i, pos, w);
        }
        println!("Uniform weights = {:.3} each", fleet.weights[0]);
        println!();
    }

    println!("=== Lesson 2: Cost Matrices ===\n");
    {
        // The cost matrix measures how much "work" it takes to move mass
        // between two distributions. Here we use cross_cost_matrix.
        let source = AgentDistribution::uniform(vec![
            vec![0.0], vec![1.0],
        ]);
        let target = AgentDistribution::uniform(vec![
            vec![0.5], vec![1.5],
        ]);
        let cost = source.cross_cost_matrix(&target);
        println!("Source positions: {:?}", source.positions);
        println!("Target positions: {:?}", target.positions);
        println!("Cost matrix (|x_i - y_j|):");
        for (i, row) in cost.iter().enumerate() {
            println!("  from {}: {:?}", i, row);
        }
        println!();
    }

    println!("=== Lesson 3: Wasserstein Distance ===\n");
    {
        // W1 = earth mover's distance: minimum cost to transform one distribution into another
        let cost = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];
        let mu = vec![0.5, 0.3, 0.2];
        let nu = vec![0.2, 0.3, 0.5];

        let w1 = OptimalTransport::wasserstein_1(&cost, &mu, &nu);
        let w2sq = OptimalTransport::wasserstein_2_squared(&cost, &mu, &nu);
        let w2 = OptimalTransport::wasserstein_2(&cost, &mu, &nu);

        println!("Distributions: mu={:?}, nu={:?}", mu, nu);
        println!("W1 distance: {:.4} (earth mover's)", w1);
        println!("W2² distance: {:.4}", w2sq);
        println!("W2 distance: {:.4}", w2);
        println!("W1 ≤ W2 always holds (W1 = {:.4} ≤ {:.4})", w1, w2);
        println!();
    }

    println!("=== Lesson 4: Sinkhorn Algorithm ===\n");
    {
        // Sinkhorn adds entropy regularization for faster, smoother transport
        let cost = vec![
            vec![0.0, 1.0, 3.0],
            vec![1.0, 0.0, 1.0],
            vec![3.0, 1.0, 0.0],
        ];
        let mu = vec![1.0/3.0; 3];
        let nu = vec![1.0/3.0; 3];

        let sinkhorn = SinkhornSolver::new(0.1); // regularization ε
        let plan = sinkhorn.solve(&cost, &mu, &nu);
        let total_cost = SinkhornSolver::transport_cost(&plan, &cost);

        println!("Regularized transport plan (ε=0.1):");
        for (i, row) in plan.iter().enumerate() {
            println!("  row {}: {:.4?}", i, row);
        }
        println!("Total transport cost: {:.4}", total_cost);
        println!("Sinkhorn gives a fuzzy transport plan (all entries > 0)");
        println!();
    }

    println!("=== Lesson 5: Agent Matching via Optimal Assignment ===\n");
    {
        // Match agents in source distribution to target positions optimally
        let agents = AgentDistribution::uniform(vec![
            vec![0.0], vec![5.0], vec![10.0],
        ]);
        let targets = AgentDistribution::uniform(vec![
            vec![1.0], vec![4.0], vec![11.0],
        ]);

        let assignment = agents.optimal_assignment(&targets);
        println!("Agents: {:?}", agents.positions);
        println!("Targets: {:?}", targets.positions);
        println!("Optimal transport assignment:");
        for (i, row) in assignment.iter().enumerate() {
            println!("  agent {} → {:?}", i, row);
        }

        let dist = agents.wasserstein_distance(&targets);
        println!("Wasserstein distance: {:.4}", dist);
        println!();
    }

    println!("=== Lesson 6: Spreading Agent Distributions ===\n");
    {
        // Spread: increase uncertainty around each agent position
        let tight = AgentDistribution::uniform(vec![vec![0.0], vec![1.0]]);
        let spread = tight.spread(2.0);

        println!("Original positions: {:?}", tight.positions);
        println!("Spread positions (factor=2.0): {:?}", spread.positions);
        println!("Spread adds Gaussian noise → diversity in fleet exploration");
        println!();
    }

    println!("=== Lesson 7: JKO Gradient Flow ===\n");
    {
        // JKO (Jordan-Kinderlehrer-Otto) scheme: gradient descent in
        // Wasserstein space. Moves distribution toward minimum energy.
        let initial = AgentDistribution::uniform(vec![
            vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0],
        ]);

        let jko = JKOScheme::new(0.1, 10); // dt=0.1, 10 steps
        let trajectory = jko.flow_to_origin(&initial);

        println!("JKO gradient flow toward origin:");
        println!("  Step 0: {} agents", trajectory[0].len());
        for (step, dist) in trajectory.iter().enumerate().take(6) {
            let mean: f64 = dist.positions.iter().map(|p| p[0]).sum::<f64>() / dist.len() as f64;
            println!("  Step {:2}: mean position = {:.4}", step, mean);
        }
        println!("  Distribution contracts toward origin (minimum energy)");
        println!();
    }

    println!("=== Lesson 8: Wasserstein Barycenters ===\n");
    {
        // Barycenter = "average" distribution in Wasserstein space
        // The Fréchet mean under W2 distance
        let cost = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];
        let mu1 = vec![0.8, 0.1, 0.1]; // concentrated at point 0
        let mu2 = vec![0.1, 0.1, 0.8]; // concentrated at point 2

        let cost_refs: Vec<Vec<f64>> = cost.clone();
        let bary = OptimalTransport::barycenter(
            &[(mu1.as_slice(), &cost_refs), (mu2.as_slice(), &cost_refs)],
            &[0.5, 0.5],
            10,
        );
        println!("Distribution 1 (left):  {:?}", mu1);
        println!("Distribution 2 (right): {:?}", mu2);
        println!("Barycenter (50/50 blend): {:?}", bary);
        println!("Barycenter balances both — mass at center point should increase");
        println!();
    }

    println!("Tutorial complete! Key takeaway:");
    println!("Optimal transport gives fleet agents a geometry — they can measure");
    println!("distances, find optimal paths, flow toward targets, and compute");
    println!("consensus via Wasserstein barycenters.");
}
