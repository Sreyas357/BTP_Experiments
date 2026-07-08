"""Run congestion MDP experiments across destination shapes, grid sizes, and stochasticity.

Usage:
  python src/run_congestion_experiments.py

This script is intentionally self-contained and data-driven:
- Destination layouts are defined once and selected by name.
- Experiment grid is generated from simple lists of sizes/shapes/noise levels.
- Each experiment runs in its own process with a hard timeout (default: 100s).
- Multiple experiments can run in parallel via `--jobs`.

Notes on stochasticity:
- In `CongestionExample`, when `stochastic=True`, the intended move happens with
  probability `p_correct` and the remaining probability is spread uniformly over
  other neighbors.
- If you think of "movement randomness" as noise probability, then:
    noise = 1 - p_correct
  So noise=0.05 => p_correct=0.95, noise=0.1 => p_correct=0.9.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import multiprocessing as mp


# Ensure repo-root modules (examples/, solvers/) are importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.congestion import CongestionExample  # noqa: E402
from examples.congestion_destinations import DESTINATION_SHAPES  # noqa: E402
from solvers.convex_opt import FusionMDPSolver  # noqa: E402
from src.helper_files.congestion_policy_io import metadata_from_spec, save_policy_json  # noqa: E402


@dataclass(frozen=True)
class ExperimentSpec:
    rows: int
    cols: int
    dest_shape: str
    horizon: int
    gamma: float
    capacity: float
    move_cost: float
    congestion_penalty: float
    stochastic: bool
    p_correct: float

    @property
    def noise(self) -> float | None:
        return None if not self.stochastic else (1.0 - self.p_correct)

    @property
    def grid_label(self) -> str:
        return f"{self.rows}x{self.cols}"

    @property
    def stoch_label(self) -> str:
        if not self.stochastic:
            return "off"
        return f"on(p={self.noise:.2f})"


def _worker_run_experiment(
    spec: ExperimentSpec,
    q: Any,
    mosek_threads: int | None = None,
    precision: float | None = None,
    policy_path: str | None = None,
) -> None:
    """Run a single experiment in a child process and report result via queue."""

    try:
        dest_fn = DESTINATION_SHAPES.get(spec.dest_shape)
        if dest_fn is None:
            raise ValueError(f"Unknown dest_shape: {spec.dest_shape}")

        t0 = time.perf_counter()

        dest_cells = dest_fn(spec.rows, spec.cols)
        example = CongestionExample(
            rows=spec.rows,
            cols=spec.cols,
            dest_cells=dest_cells,
            stochastic=spec.stochastic,
            p_correct=spec.p_correct,
            capacity=spec.capacity,
            move_cost=spec.move_cost,
            congestion_penalty=spec.congestion_penalty,
        )
        mdp = example.generate_congestion_mdp()

        t1 = time.perf_counter()
        solver = FusionMDPSolver(
            mdp,
            horizon=spec.horizon,
            gamma=spec.gamma,
            mosek_num_threads=mosek_threads,
            precision=precision,
        )
        _policy, objective = solver.solve()
        t2 = time.perf_counter()

        if policy_path is not None:
            metadata = metadata_from_spec(
                spec,
                extra={
                    "dest_cells": dest_cells,
                    "initial_dist": example.initial_dist,
                    "time_taken": t2 - t1,
                },
            )
            save_policy_json(policy_path, metadata=metadata, policy=_policy)

        q.put(
            {
                "status": "ok",
                "objective": float(objective),
                "time_s": t2 - t0,
                "policy_path": policy_path,
            }
        )

    except BaseException as e:  # includes KeyboardInterrupt in the worker
        q.put({"status": "error", "error": f"{type(e).__name__}: {e}"})


def _run_with_timeout(
    spec: ExperimentSpec,
    timeout_s: float,
    mosek_threads: int | None = None,
    precision: float | None = None,
    policy_path: str | None = None,
) -> dict[str, Any]:
    """Run one experiment with a hard wall-clock timeout.

    Implemented using a separate process so we can terminate long MOSEK runs.
    """

    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=_worker_run_experiment,
        args=(spec, q, mosek_threads, precision, policy_path),
    )

    wall_start = time.perf_counter()
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        p.terminate()
        p.join(timeout=5.0)
        wall_end = time.perf_counter()
        return {
            "status": "timeout",
            "time_s": wall_end - wall_start,
        }

    wall_end = time.perf_counter()

    result: dict[str, Any]
    try:
        result = q.get_nowait()
    except Exception:
        result = {"status": "error", "error": "Worker exited without returning a result"}

    if "time_s" not in result:
        result["time_s"] = wall_end - wall_start
    return result


def _parse_sizes(values: list[str]) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for v in values:
        s = v.lower().replace(" ", "")
        if "x" not in s:
            raise argparse.ArgumentTypeError(f"Invalid size '{v}'. Use ROWSxCOLS, e.g. 10x10")
        r_str, c_str = s.split("x", 1)
        rows = int(r_str)
        cols = int(c_str)
        if rows <= 0 or cols <= 0:
            raise argparse.ArgumentTypeError(f"Invalid grid size '{v}'. Rows/cols must be positive")
        sizes.append((rows, cols))
    return sizes


def _format_float(v: float | None, width: int, prec: int) -> str:
    if v is None:
        return "".rjust(width)
    return f"{v:{width}.{prec}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run congestion experiments with timing + timeout")

    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["5x5", "10x10", "15x15"],
        help="Grid sizes as ROWSxCOLS (default: 20x20 40x40 80x80 )",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["iitb", "hello_world", "smiley"],
        choices=list(DESTINATION_SHAPES.keys()),
        help="Destination shapes to run",
    )

    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        required=True,
        help="Time horizon(s). Provide one value (broadcast to all sizes) or one per --sizes entry.",
    )
    
    
    parser.add_argument("--gamma", type=float, default=1.0)

    parser.add_argument(
        "--capacity",
        type=float,
        default=-1.0,
        help="Capacity per cell. If <= 0, defaults to 1/rows (as requested)",
    )
    parser.add_argument("--move-cost", type=float, default=1.0)
    parser.add_argument("--congestion-penalty", type=float, default=5.0)

    parser.add_argument(
        "--noise",
        nargs="+",
        type=float,
        default=[0.05, 0.10],
        help="Movement noise probabilities when stochastic is ON (default: 0.05 0.10)",
    )
    parser.add_argument(
        "--randomized",
        action="store_true",
        help="If set, run only randomized (stochastic=on) experiments",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=100.0,
        help="Per-experiment wall-clock timeout in seconds (default: 100)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=mp.cpu_count(),
        help="Number of experiments to run in parallel (default: CPU count)",
    )
    parser.add_argument(
        "--mosek-threads",
        type=int,
        default=0,
        help="MOSEK threads per experiment process. Use 0 to keep MOSEK default.",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=1e-5,
        help=(
            "MOSEK interior-point tolerance used by FusionMDPSolver. "
            "Larger values are faster but less accurate (default: 1e-5)."
        ),
    )

    parser.add_argument(
        "--policy-dir",
        type=str,
        default="artifacts/congestion_policies",
        help="Directory to write solved policies (JSON)",
    )

    args = parser.parse_args(argv)

    if not (0 < args.gamma <= 1.0):
        raise SystemExit("--gamma must be in (0, 1]")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    if args.mosek_threads < 0:
        raise SystemExit("--mosek-threads must be >= 0")
    if args.precision <= 0:
        raise SystemExit("--precision must be > 0")

    sizes = _parse_sizes(args.sizes)
    shapes: list[str] = list(args.shapes)

    # Horizons can be provided as:
    # - a single value (applied to all sizes)
    # - one value per size
    raw_horizons = list(args.horizons)
    if any(h <= 0 for h in raw_horizons):
        raise SystemExit("--horizons values must be > 0")

    if len(raw_horizons) == 1:
        horizons = [int(raw_horizons[0])] * len(sizes)
    elif len(raw_horizons) == len(sizes):
        horizons = [int(h) for h in raw_horizons]
    else:
        raise SystemExit(
            "--horizons must have length 1 or match --sizes length. "
            f"Got horizons={len(raw_horizons)}, sizes={len(sizes)}"
        )

    noise_levels = []
    for n in args.noise:
        if not (0.0 < n < 1.0):
            raise SystemExit("--noise values must be in (0, 1)")
        noise_levels.append(float(n))

    experiments: list[ExperimentSpec] = []
    
    for horizon, (rows, cols) in zip(horizons, sizes):
        
        cap = (1.0 / rows) if args.capacity <= 0 else float(args.capacity)
        for shape in shapes:
            if not args.randomized:
                experiments.append(
                    ExperimentSpec(
                        rows=rows,
                        cols=cols,
                        dest_shape=shape,
                        horizon=horizon,
                        gamma=args.gamma,
                        capacity=cap,
                        move_cost=args.move_cost,
                        congestion_penalty=args.congestion_penalty,
                        stochastic=False,
                        p_correct=1.0,
                    )
                )

            else:
                for noise in noise_levels:
                    experiments.append(
                        ExperimentSpec(
                            rows=rows,
                            cols=cols,
                            dest_shape=shape,
                            horizon=horizon,
                            gamma=args.gamma,
                            capacity=cap,
                            move_cost=args.move_cost,
                            congestion_penalty=args.congestion_penalty,
                            stochastic=True,
                            p_correct=1.0 - noise,
                        )
                    )

    header = (
        f"{'grid':>7} {'H':>4}  {'shape':>12}  {'stoch':>9}  "
        f"{'objective':>11}  {'time_s':>7}  {'status':>7}"
    )
    print(header)
    print("-" * len(header))

    total_start = time.perf_counter()

    n_ok = n_timeout = n_error = 0
    times: list[float] = []

    policy_dir = Path(args.policy_dir)

    ordered_results: dict[int, tuple[ExperimentSpec, dict[str, Any]]] = {}
    mosek_threads = None if args.mosek_threads == 0 else int(args.mosek_threads)
    precision_tag = f"p{args.precision:g}"
    if int(args.jobs) == 1 or len(experiments) == 1:
        for i, spec in enumerate(experiments, start=1):
            stoch_tag = "det" if not spec.stochastic else f"noise{spec.noise:.2f}"
            policy_path = policy_dir / (
                f"policy_{spec.rows}x{spec.cols}_{spec.dest_shape}_{stoch_tag}_H{spec.horizon}_{precision_tag}_{i:04d}.json"
            )
            result = _run_with_timeout(
                spec,
                float(args.timeout),
                mosek_threads,
                float(args.precision),
                str(policy_path),
            )
            ordered_results[i] = (spec, result)
    else:
        with ThreadPoolExecutor(max_workers=int(args.jobs)) as executor:
            future_to_item = {
                executor.submit(
                    _run_with_timeout,
                    spec,
                    float(args.timeout),
                    mosek_threads,
                    float(args.precision),
                    str(
                        policy_dir
                        / (
                            f"policy_{spec.rows}x{spec.cols}_{spec.dest_shape}_{('det' if not spec.stochastic else f'noise{spec.noise:.2f}')}_H{spec.horizon}_{precision_tag}_{i:04d}.json"
                        )
                    ),
                ): (i, spec)
                for i, spec in enumerate(experiments, start=1)
            }

            completed = 0
            for future in as_completed(future_to_item):
                i, spec = future_to_item[future]
                try:
                    result = future.result()
                except BaseException as e:
                    result = {"status": "error", "error": f"{type(e).__name__}: {e}"}

                ordered_results[i] = (spec, result)
                completed += 1

                # Light progress indicator for long runs
                if completed % 10 == 0 and completed != len(experiments):
                    elapsed = time.perf_counter() - total_start
                    print(f"-- progress: {completed}/{len(experiments)} done, elapsed={elapsed:.1f}s --")

    for i in range(1, len(experiments) + 1):
        spec, result = ordered_results[i]

        status = result.get("status", "error")
        objective = result.get("objective")
        time_s = result.get("time_s")

        if status == "ok":
            n_ok += 1
            if isinstance(time_s, (int, float)):
                times.append(float(time_s))
        elif status == "timeout":
            n_timeout += 1
            if isinstance(time_s, (int, float)):
                times.append(float(time_s))
        else:
            n_error += 1

        obj_str = _format_float(objective if isinstance(objective, (int, float)) else None, 11, 4)
        time_str = _format_float(time_s if isinstance(time_s, (int, float)) else None, 7, 3)

        print(
            f"{spec.grid_label:>7} {spec.horizon:>4d}  {spec.dest_shape:>12}  {spec.stoch_label:>9}  "
            f"{obj_str}  {time_str}  {status:>7}"
        )

        if status == "error" and "error" in result:
            print(f"{'':>8}  {'':>12}  {'':>10}  {'':>12}  {'':>8}  {'':>8}  error: {result['error']}")

    total_end = time.perf_counter()

    def _stats(xs: list[float]) -> tuple[float, float, float] | None:
        if not xs:
            return None
        xs_sorted = sorted(xs)
        return xs_sorted[0], xs_sorted[len(xs_sorted) // 2], xs_sorted[-1]

    time_stats = _stats(times)

    print("\nSummary")
    print("-------")
    print(f"experiments: {len(experiments)}")
    print(f"ok={n_ok}, timeout={n_timeout}, error={n_error}")
    print(f"total_runtime_s: {total_end - total_start:.3f}")

    if time_stats is not None:
        mn, med, mx = time_stats
        print(f"time_s (min/med/max): {mn:.3f} / {med:.3f} / {mx:.3f}")

    return 0 if n_error == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
