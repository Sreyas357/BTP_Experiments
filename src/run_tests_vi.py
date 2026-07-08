import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.path_concentration import Warehouse_path_concentration_mdp
from solvers.value_iteration import MaxLinearValueIterationSolver
from src.helper_files.congestion_policy_io import save_policy_json


def _policy_rollout_from_vi(solver, mdp, horizon: int):
    mu = dict(mdp.initial_dist)
    policy = []
    states = list(mdp.states)

    for t in range(horizon):
        pi_det = solver.best_policy_for_distribution(mu, t)
        policy.append({s: {pi_det[s]: 1.0} for s in states})

        mu_next = {s: 0.0 for s in states}
        for s in states:
            mass = mu[s]
            if mass <= 0.0:
                continue
            a = pi_det[s]
            for s_next in states:
                p = mdp.transition_prob.get((s, a, s_next), 0.0)
                if p > 0.0:
                    mu_next[s_next] += mass * p
        mu = mu_next

    return policy


def _cell_from_state(state_name: str):
    _, i, j = state_name.split("_")
    return [int(i), int(j)]


def _fmt_num(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def _build_output_name(method: str, args) -> str:
    mode = "stoch" if args.stochastic else "det"
    n_pen = args.num_random_penalty_states
    n_pen_tag = "auto" if n_pen is None else str(n_pen)
    return (
        f"pathconc_r{args.rows}c{args.cols}_h{args.horizon}"
        f"_g{_fmt_num(args.gamma)}_tp{_fmt_num(args.time_penalty)}"
        f"_cp{_fmt_num(args.concentration_penalty)}_pc{_fmt_num(args.penalty_states_cost)}"
        f"_np{n_pen_tag}_{mode}_{method}.json"
    )


def _run_case(name: str, mdp, args, out_path: Path, use_pbvi: bool, penalty_cells):
    kwargs = dict(
        mdp=mdp,
        horizon=args.horizon,
        gamma=args.gamma,
        prune=True,
        policy_batch_size=64,
    )
    if use_pbvi:
        kwargs.update(
            max_vectors=args.pbvi_max_vectors,
            n_belief_points=args.pbvi_belief_points,
            seed=args.seed,
        )

    solver = MaxLinearValueIterationSolver(**kwargs)
    started = time.perf_counter()
    gamma_sets = solver.solve()

    elapsed = time.perf_counter() - started
    v0 = solver.value(mdp.initial_dist, t=0)
    policy = _policy_rollout_from_vi(solver, mdp, args.horizon)

    payload_meta = {
        "example": "path_concentration",
        "method": name,
        "rows": args.rows,
        "cols": args.cols,
        "dest_cells": [[args.cols - 1, j] for j in range(args.rows)],
        "penalty_cells": penalty_cells,
        "initial_dist": mdp.initial_dist,
        "stochastic": args.stochastic,
        "p_correct": args.p_correct,
        "horizon": args.horizon,
        "gamma": args.gamma,
    }
    save_policy_json(out_path, metadata=payload_meta, policy=policy)

    print(
        f"[{name}] solved in {elapsed:.2f}s | V0={v0:.6f} | "
        f"|Gamma0|={len(gamma_sets[0])} | saved={out_path}"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run path concentration VI (exact + PBVI) and save policies")
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--time-penalty", type=float, default=1.0)
    p.add_argument("--concentration-penalty", type=float, default=2.0)
    p.add_argument("--penalty-states-cost", type=float, default=5.0)
    p.add_argument("--num-random-penalty-states", type=int, default=None)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--p-correct", type=float, default=0.8)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--pbvi-max-vectors", type=int, default=100)
    p.add_argument("--pbvi-belief-points", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="artifacts/vi_policies")

    # Internal worker mode used by subprocess timeout wrapper.
    p.add_argument("--worker", action="store_true")
    p.add_argument("--method", choices=["vi_exact", "vi_pbvi"], default=None)
    p.add_argument("--output", type=str, default=None)
    return p


def _run_worker(args) -> int:
    if args.method is None or args.output is None:
        raise ValueError("--method and --output are required in --worker mode")

    example = Warehouse_path_concentration_mdp(
        rows=args.rows,
        cols=args.cols,
        time_penalty=args.time_penalty,
        concentration_penalty=args.concentration_penalty,
        penalty_states_cost=args.penalty_states_cost,
        num_random_penalty_states=args.num_random_penalty_states,
        is_stochastic=args.stochastic,
        p_correct=args.p_correct,
        random_seed=args.seed,
    )
    mdp, _, _, _ = example.generate_mdp()
    penalty_cells = [_cell_from_state(s) for s in example.selected_penalty_states]

    _run_case(
        args.method,
        mdp,
        args,
        Path(args.output),
        use_pbvi=(args.method == "vi_pbvi"),
        penalty_cells=penalty_cells,
    )
    return 0


def _build_worker_cmd(args, method: str, out_path: Path):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--method",
        method,
        "--output",
        str(out_path),
        "--rows",
        str(args.rows),
        "--cols",
        str(args.cols),
        "--horizon",
        str(args.horizon),
        "--gamma",
        str(args.gamma),
        "--time-penalty",
        str(args.time_penalty),
        "--concentration-penalty",
        str(args.concentration_penalty),
        "--penalty-states-cost",
        str(args.penalty_states_cost),
        "--p-correct",
        str(args.p_correct),
        "--pbvi-max-vectors",
        str(args.pbvi_max_vectors),
        "--pbvi-belief-points",
        str(args.pbvi_belief_points),
        "--seed",
        str(args.seed),
    ]
    if args.stochastic:
        cmd.append("--stochastic")
    if args.num_random_penalty_states is not None:
        cmd.extend(["--num-random-penalty-states", str(args.num_random_penalty_states)])
    return cmd


def main():
    args = _build_parser().parse_args()

    if args.worker:
        return _run_worker(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for method in ["vi_pbvi"]:
        out_path = out_dir / _build_output_name(method, args)
        cmd = _build_worker_cmd(args, method, out_path)
        try:
            subprocess.run(cmd, check=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"[{method}] timeout after {args.timeout}s")
        except subprocess.CalledProcessError as exc:
            if exc.returncode < 0:
                print(
                    f"[{method}] failed: process killed by signal {-exc.returncode} "
                    "(often OOM). Try smaller rows/cols or lower complexity."
                )
            else:
                print(f"[{method}] failed with exit code {exc.returncode}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())