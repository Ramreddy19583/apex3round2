# Enhanced Savings Router (VRPTW Solver)

## 📌 Overview
The `EnhancedSavingsRouter` is a high-performance metaheuristic solver designed for the **Vehicle Routing Problem with Time Windows (VRPTW)**. The solver aims to minimize total route cost while satisfying vehicle capacity constraints and strict customer time windows.

It employs a hybrid architecture: **Multi-stage Construction $\rightarrow$ Adaptive Large Neighborhood Search (ALNS) $\rightarrow$ Local Search Refinement**.

---

## 🚀 Key Features

### 1. Robust Construction Heuristics
To ensure a valid starting solution regardless of instance difficulty, the router attempts four construction strategies in descending order of quality:
- **Regret-k Insertion:** Prioritizes customers who have a large cost difference between their best and second-best insertion points.
- **Greedy Append:** An urgency-based approach sorting customers by time-window closing times.
- **Sweep Construction:** A geometric approach clustering customers by their polar angle relative to the depot.
- **Clarke-Wright Savings:** Merges routes based on the distance saved by combining two separate trips into one.

### 2. Adaptive Large Neighborhood Search (ALNS)
The solver uses a "Ruin and Repair" loop to explore the search space and escape local optima:
- **Ruin Strategies:**
    - **Shaw Ruin:** Removes customers based on similarity (distance, time windows, and demand).
    - **Worst-Cost Ruin:** Targets customers contributing the highest marginal cost to the current routes.
    - **Cluster Ruin:** Removes a geographically concentrated group of customers.
- **Repair Strategy:** Uses **Regret Insertion** to optimally re-integrate removed customers into the plan.
- **Simulated Annealing:** Accepts slightly worse solutions based on a cooling temperature to ensure global exploration.

### 3. Local Search Refinement (`Fast Descend`)
Once a global structure is found, the solver applies iterative hill-climbing to polish the routes:
- **Relocate Moves:** Shifts single nodes or segments between different positions or vehicles.
- **2-Opt:** Reverses path segments to eliminate route crossings and reduce distance.

---

## 🛠 Technical Specifications

- **Time Management:** The solver is deadline-aware. It monitors execution time and utilizes a `RESERVE` buffer to ensure the best-found solution is submitted before the time limit expires.
- **Complexity Optimization:** 
    - Precomputes a **Shaw Distance Matrix** for $O(1)$ similarity lookups.
    - Utilizes **Neighbour Lists** to limit the search space for local search, preventing exponential slowdown on large instances.
- **Deterministic RNG:** Uses a seeded random number generator (`Rng`) based on the instance digest for reproducible results.

## 📦 Dependencies
- `adapter`: Routing utility functions (`route_cost`, `simulate_route`, `assign_vehicles`).
- `benchkit`: Deterministic Random Number Generation.
- `data`: Problem instance definitions.
