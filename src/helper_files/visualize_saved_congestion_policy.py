"""Visualize a saved congestion policy.

Example:
  python3 src/visualize_saved_congestion_policy.py --policy artifacts/congestion_policies/policy_0001.json

The visualization reconstructs the congestion MDP from the saved metadata:
(grid size, stochasticity, destination cells, etc.), then simulates the policy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Ensure repo-root modules (examples/, solvers/) are importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.helper_files.congestion_policy_io import load_policy_json  # noqa: E402
from src.helper_files.visualize_policy import animate_congestion_policy  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Animate a saved congestion policy")
    p.add_argument("--policy", required=True, help="Path to saved policy JSON")
    p.add_argument("--interval-ms", type=int, default=250)
    p.add_argument("--horizon", type=int, default=None, help="Override horizon for animation")

    args = p.parse_args(argv)

    payload = load_policy_json(args.policy)
    meta = payload.get("metadata", {})
    policy = payload["policy"]

    rows = int(meta["rows"])
    cols = int(meta["cols"])
    dest_cells = [tuple(x) for x in meta["dest_cells"]]

    initial_dist = meta.get("initial_dist")
    stochastic = bool(meta.get("stochastic", False))
    p_correct = float(meta.get("p_correct", 1.0))

    horizon = args.horizon
    if horizon is None:
        horizon = int(meta.get("horizon", len(policy)))

    animate_congestion_policy(
        rows=rows,
        cols=cols,
        initial_dist=initial_dist,
        dest_cells=dest_cells,
        policy=policy,
        horizon=horizon,
        stochastic=stochastic,
        p_correct=p_correct,
        interval_ms=args.interval_ms,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
