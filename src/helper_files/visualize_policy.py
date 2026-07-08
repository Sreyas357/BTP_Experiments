import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numpy as np

# Ensure repo-root modules (examples/, solvers/) are importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.congestion import CongestionExample
from solvers.convex_opt import FusionMDPSolver


def parse_state_name(state_name: str):
    _, i, j = state_name.split("_")
    return int(i), int(j)


def infer_grid_size(states):
    max_i = 0
    max_j = 0
    for s in states:
        i, j = parse_state_name(s)
        if i > max_i:
            max_i = i
        if j > max_j:
            max_j = j
    return max_j + 1, max_i + 1  # rows, cols


def rollout_occupancy(mdp, policy, horizon=None):
    states = mdp.states
    state_idx = {s: i for i, s in enumerate(states)}

    if horizon is None:
        horizon = len(policy)
    horizon = min(horizon, len(policy))

    S = len(states)
    mu = np.zeros((horizon + 1, S), dtype=float)
    for s, p in mdp.initial_dist.items():
        mu[0, state_idx[s]] = float(p)

    trans_by_sa = {}
    for (s, a, s_next), p in mdp.transition_prob.items():
        if p > 0.0:
            trans_by_sa.setdefault((s, a), []).append((s_next, float(p)))

    for t in range(horizon):
        for s in states:
            i = state_idx[s]
            mass = mu[t, i]
            if mass <= 0.0:
                continue

            action_dist = policy[t].get(s, {})
            for a, pi_sa in action_dist.items():
                if a not in mdp.state_action_map[s] or pi_sa <= 0.0:
                    continue
                flow = mass * float(pi_sa)
                if flow <= 0.0:
                    continue

                for s_next, p in trans_by_sa.get((s, a), []):
                    mu[t + 1, state_idx[s_next]] += flow * p

    return mu


def _grid_from_mu(mu_t, states, rows, cols):
    grid = np.zeros((rows, cols), dtype=float)
    for idx, s in enumerate(states):
        i, j = parse_state_name(s)
        grid[j, i] = mu_t[idx]
    return grid


def animate_grid_occupancy(
    mdp,
    policy,
    horizon=None,
    interval_ms=250,
    dest_cells=None,
    penalty_cells=None,
):
    rows, cols = infer_grid_size(mdp.states)
    mu = rollout_occupancy(mdp, policy, horizon=horizon)

    dest_set = set(dest_cells or [])
    penalty_set = set(penalty_cells or [])
    dest_states = {f"s_{i}_{j}" for (i, j) in dest_set}
    penalty_states = {f"s_{i}_{j}" for (i, j) in penalty_set}
    states = mdp.states
    state_idx = {s: i for i, s in enumerate(states)}

    grids = [_grid_from_mu(mu[t], states, rows, cols) for t in range(mu.shape[0])]
    vmax = max(float(np.max(g)) for g in grids)
    if vmax <= 1e-12:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    img = ax.imshow(
        grids[0],
        origin="lower",
        extent=(0, cols, 0, rows),
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Occupancy mass")

    ax.set_title("MDP Occupancy Animation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.grid(color="white", linestyle="-", linewidth=0.5, alpha=0.25)

    if dest_set:
        # Destination cells are plotted at cell centers while axes show
        # corner coordinates, so (0, 0) is the true bottom-left corner.
        xs = [i + 0.5 for (i, _) in dest_set]
        ys = [j + 0.5 for (_, j) in dest_set]
        ax.scatter(xs, ys, marker="s", s=120, facecolors="none", edgecolors="red", linewidths=1.2)

    if penalty_set:
        xs = [i + 0.5 for (i, _) in penalty_set]
        ys = [j + 0.5 for (_, j) in penalty_set]
        ax.scatter(xs, ys, marker="x", s=90, c="orange", linewidths=1.8)

    status = ax.text(
        0.02,
        1.02,
        "",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
    )

    is_paused = {"value": False}

    def update(frame_t):
        img.set_data(grids[frame_t])
        if dest_states and penalty_states:
            dest_mass = sum(mu[frame_t, state_idx[s]] for s in dest_states if s in state_idx)
            penalty_mass = sum(mu[frame_t, state_idx[s]] for s in penalty_states if s in state_idx)
            status.set_text(
                f"t = {frame_t} | dest mass = {dest_mass:.4f} | penalty mass = {penalty_mass:.4f}"
            )
        elif dest_states:
            dest_mass = sum(mu[frame_t, state_idx[s]] for s in dest_states if s in state_idx)
            status.set_text(f"t = {frame_t} | dest mass = {dest_mass:.4f}")
        else:
            status.set_text(f"t = {frame_t}")
        return img, status

    anim = FuncAnimation(
        fig,
        update,
        frames=len(grids),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    # Pause/Play button.
    btn_ax = fig.add_axes([0.78, 0.01, 0.16, 0.06])
    btn = Button(btn_ax, "Pause")

    def toggle_pause(_event=None):
        if is_paused["value"]:
            anim.event_source.start()
            is_paused["value"] = False
            btn.label.set_text("Pause")
        else:
            anim.event_source.stop()
            is_paused["value"] = True
            btn.label.set_text("Play")
        fig.canvas.draw_idle()

    btn.on_clicked(toggle_pause)

    # Keyboard shortcut: Space to pause/play.
    def on_key_press(event):
        if event.key == " ":
            toggle_pause()

    fig.canvas.mpl_connect("key_press_event", on_key_press)

    fig._anim_ref = anim
    plt.tight_layout()
    plt.show()


def animate_congestion_policy(
    *,
    rows: int,
    cols: int,
    initial_dist: dict[str, float] | None,
    dest_cells: list[tuple[int, int]],
    policy,
    horizon: int,
    stochastic: bool,
    p_correct: float,
    interval_ms: int = 250,
    capacity: float | None = None,
    move_cost: float = 1.0,
    congestion_penalty: float = 5.0,
):
    """Reconstruct the congestion MDP from grid + params and animate a policy.

    This matches the requested API surface: provide grid size, initial distribution,
    destination set, policy, horizon, and stochasticity parameters.
    """

    if capacity is None:
        capacity = 1.0 / rows

    example = CongestionExample(
        rows=rows,
        cols=cols,
        dest_cells=dest_cells,
        initial_dist=initial_dist,
        stochastic=stochastic,
        p_correct=p_correct,
        capacity=capacity,
        move_cost=move_cost,
        congestion_penalty=congestion_penalty,
    )
    mdp = example.generate_congestion_mdp()
    animate_grid_occupancy(
        mdp=mdp,
        policy=policy,
        horizon=horizon,
        interval_ms=interval_ms,
        dest_cells=dest_cells,
    )


if __name__ == "__main__":
    # This file is now meant as a library module.
    # Use `src/visualize_saved_congestion_policy.py` to animate saved policies.
    pass
