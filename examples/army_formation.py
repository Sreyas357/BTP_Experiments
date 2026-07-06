import numpy as np

from mdp import MDP
from mdp.reward import LinearReward, MaxLinearReward


def generate_army_formation_mdp(
    rows: int = 3,
    cols: int = 4,
    window_size: int = 2,
    # Reward weights (mirroring the report's notation loosely)
    alpha_progress: float = 1.0,
    beta_formation: float = 2.0,
    danger_cost: float = 3.0,
    time_cost: float = 1.0,
    # Switch dynamics
    switch_split_prob: float = 0.5,
):
    """
    Scaled-down version of the report's "Army Formation with Obstacle Avoidance".

    Why scaled down?
    --------------
    `MaxLinearValueIterationSolver` enumerates all deterministic state->action policies.
    A 10x6 grid would be intractable. This generator keeps |S| small so VI is runnable.

    State space
    ----------
    Grid cells: s_i_j for i in [0..cols-1], j in [0..rows-1].

    Actions
    -------
    - "right": deterministic move to (i+1,j) (or stay at boundary)
    - "switch": move between "corridors" (rows). In the middle row, this can split
      probability mass up/down to keep action count small per state.
    - "stay": terminal column only

    Reward (max-linear, VI-compatible)
    ---------------------------------
    We implement the same *structure* as in the report:

    - Progress:   alpha * r_prog^T mu, where r_prog(i,j) = i/(cols-1)
    - Formation:  beta * max_{W in windows} 1_W^T mu
    - Danger:     linear penalty -danger_cost * phi_D(mu)
    - Time cost:  linear penalty -time_cost * mass_not_in_goal

    Note: the report uses a thresholded danger penalty of the form -C*max(0,phi_D-kappa),
    which is concave piecewise-linear. This VI solver is built for max-linear (convex)
    rewards, so we use a linear danger penalty here to preserve VI compatibility.

    Returns
    -------
    mdp : MDP
    info : dict with keys: goal_states, danger_states, windows
    """

    if rows < 2 or cols < 2:
        raise ValueError("rows and cols must be >= 2")
    if not (1 <= window_size <= min(rows, cols)):
        raise ValueError("window_size must be in [1, min(rows, cols)]")
    if cols == 1:
        raise ValueError("cols must be > 1 to define progress")
    if not (0.0 < switch_split_prob < 1.0):
        raise ValueError("switch_split_prob must be in (0, 1)")

    def state(i: int, j: int) -> str:
        return f"s_{i}_{j}"

    states = [state(i, j) for i in range(cols) for j in range(rows)]
    state_to_idx = {s: k for k, s in enumerate(states)}

    actions = ["right", "switch", "stay"]

    # --- action availability ---
    state_action_map = {}
    for i in range(cols):
        for j in range(rows):
            s = state(i, j)
            if i == cols - 1:
                state_action_map[s] = ["stay"]
            else:
                # Keep exactly 2 actions per non-terminal state to keep policy
                # enumeration manageable.
                state_action_map[s] = ["right", "switch"]

    # --- transitions ---
    transition_prob = {}
    for i in range(cols):
        for j in range(rows):
            s = state(i, j)

            for a in state_action_map[s]:
                if a == "stay":
                    transition_prob[(s, a, s)] = 1.0
                    continue

                if a == "right":
                    if i == cols - 1:
                        transition_prob[(s, a, s)] = 1.0
                    else:
                        transition_prob[(s, a, state(i + 1, j))] = 1.0
                    continue

                # a == "switch"
                if rows == 2:
                    # toggle between the two rows
                    transition_prob[(s, a, state(i, 1 - j))] = 1.0
                else:
                    mid = rows // 2
                    if j == 0:
                        transition_prob[(s, a, state(i, 1))] = 1.0
                    elif j == rows - 1:
                        transition_prob[(s, a, state(i, rows - 2))] = 1.0
                    elif j == mid:
                        # split mass up/down from the middle corridor
                        transition_prob[(s, a, state(i, j - 1))] = switch_split_prob
                        transition_prob[(s, a, state(i, j + 1))] = 1.0 - switch_split_prob
                    else:
                        # for rows > 3, move one step toward the middle
                        step = -1 if j > mid else 1
                        transition_prob[(s, a, state(i, j + step))] = 1.0

    # --- initial distribution: uniform on leftmost column ---
    initial_dist = {s: 0.0 for s in states}
    for j in range(rows):
        initial_dist[state(0, j)] = 1.0 / rows

    # --- goal states: rightmost column ---
    goal_states = {state(cols - 1, j) for j in range(rows)}

    # --- danger zones (small, staggered) ---
    # These are chosen to force a corridor switch in a small grid.
    danger_cells = []
    if cols >= 3 and rows >= 3:
        danger_cells = [(1, rows // 2), (2, 0)]  # staggered
    elif cols >= 3:
        danger_cells = [(1, 0)]
    else:
        danger_cells = [(0, 0)]

    danger_states = {state(i, j) for (i, j) in danger_cells}
    danger_indices = [state_to_idx[s] for s in danger_states]

    # --- reward components ---
    # Progress vector: i/(cols-1)
    progress_weights = []
    progress_indices = []
    for i in range(cols):
        for j in range(rows):
            s = state(i, j)
            progress_indices.append(state_to_idx[s])
            progress_weights.append(alpha_progress * (i / (cols - 1)))

    # Time cost on non-goal mass
    non_goal_indices = [
        state_to_idx[s] for s in states if s not in goal_states
    ]

    # Formation windows: all r x r sliding windows
    windows = []
    r = window_size
    for i in range(cols - r + 1):
        for j in range(rows - r + 1):
            W = [(ii, jj) for ii in range(i, i + r) for jj in range(j, j + r)]
            windows.append(W)

    window_vectors = []
    for W in windows:
        v = np.zeros(len(states), dtype=float)
        for (i, j) in W:
            v[state_to_idx[state(i, j)]] = 1.0
        window_vectors.append(v)

    reward_model = [
        # Progress term (linear)
        LinearReward(indices=progress_indices, weights=progress_weights, offset=0.0),
        # Formation cohesion (max over window mass)
        MaxLinearReward(
            vectors=window_vectors,
            offsets=[0.0] * len(window_vectors),
            weight=float(beta_formation),
        ),
        # Danger avoidance (linear penalty)
        LinearReward(
            indices=danger_indices,
            weights=[-float(danger_cost)] * len(danger_indices),
            offset=0.0,
        ),
        # Time cost (linear penalty)
        LinearReward(
            indices=non_goal_indices,
            weights=[-float(time_cost)] * len(non_goal_indices),
            offset=0.0,
        ),
    ]

    mdp = MDP(
        states=states,
        actions=actions,
        state_action_map=state_action_map,
        transition_prob=transition_prob,
        initial_dist=initial_dist,
        reward_model=reward_model,
        is_max_linear_reward=True,
        is_concave_reward=False,
    )

    info = {
        "goal_states": goal_states,
        "danger_states": danger_states,
        "windows": windows,
        "rows": rows,
        "cols": cols,
        "window_size": window_size,
        "danger_cells": danger_cells,
    }

    return mdp, info
