# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirments.txt  # note: filename is intentionally misspelled
```

Key dependencies: `numpy`, `scipy`, `cvxpy`, `mosek`, `matplotlib`, `pandas`, `pydantic`

## Running Scripts

```bash
python src/run_tests.py          # Congestion MDP benchmark using convex optimization (MOSEK)
python src/run_tests_vi.py       # Warehouse path-concentration benchmark using value iteration
python src/visualize_policy.py   # Interactive animated policy visualizer
python src/benchmark.py          # Generic timing benchmark harness
python mdp/test_mdp.py           # Basic MDP validation tests
```

## Architecture

This is a research codebase for **distributional MDP optimization** — optimizing over distributions of trajectories rather than single-state policies.

### Core Framework (`mdp/`)

- **`mdp.py`**: Pydantic `MDP` class. Stores states, actions, transition tensor `P[s, a, s']`, initial distribution, horizon `T`, discount `gamma`. Supports two reward specs: string expressions (AST-parsed) or structured `reward_model` (list of `LinearReward`/`MaxLinearReward` terms).
- **`reward.py`**: `LinearReward` (`a^T μ + b`) and `MaxLinearReward` (`weight * max_i(a_i^T μ + b_i)`) term classes.

### Solvers (`solvers/`)

Two distinct solver paradigms for different reward structures:

- **`convex_opt.py` — `FusionMDPSolver`**: Uses MOSEK Fusion API to solve concave reward MDPs as conic programs. Input: MDP with `reward_model` and `is_concave_reward=True`. Output: per-timestep action distributions (occupancy measures) and objective value.

- **`value_iteration.py` — `MaxLinearValueIterationSolver`**: Enumeration + pruning approach for max-linear rewards. Maintains a set of value function candidates (gamma set) at each timestep via Bellman recursion, pruning dominated candidates. Works with max-linear reward structures where convex optimization doesn't directly apply.

### Examples (`examples/`)

Domain-specific MDP generators: `CongestionExample` (grid-world agents, H-shaped destinations), `path_concentration.py` (2×5 warehouse), `reach_avoidance.py` (reach-avoid with stochastic slip). These instantiate `MDP` objects and are consumed by `src/` scripts.

### Key Design Choices

- The MDP state is a **joint state** of multiple agents; the "distribution" being optimized is over joint trajectories.
- `FusionMDPSolver` solves in the **occupancy measure** space (continuous LP/conic program), not via explicit policy enumeration.
- `MaxLinearValueIterationSolver` explicitly enumerates deterministic policies and prunes the value function candidates at each step — complexity grows with the gamma set size.
- Reward functions are evaluated on **state-action occupancy measures** `μ ∈ R^{|S|×|A|}`, not pointwise on individual state-action pairs.
