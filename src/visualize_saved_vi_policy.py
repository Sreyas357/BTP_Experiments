"""Visualize a saved VI path-concentration policy.

Example:
  python src/visualize_saved_vi_policy.py --policy artifacts/vi_policies/path_concentration_vi_exact.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.path_concentration import generate_warehouse_path_concentration_mdp  # noqa: E402
from src.congestion_policy_io import load_policy_json  # noqa: E402
from src.visualize_policy import animate_grid_occupancy  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Animate a saved VI policy")
    p.add_argument("--policy", required=True, help="Path to saved policy JSON")
    p.add_argument("--interval-ms", type=int, default=250)
    p.add_argument("--horizon", type=int, default=None)
    args = p.parse_args(argv)

    payload = load_policy_json(args.policy)
    meta = payload.get("metadata", {})
    policy = payload["policy"]

    rows = int(meta["rows"])
    cols = int(meta["cols"])
    horizon = int(meta.get("horizon", len(policy))) if args.horizon is None else args.horizon

    dest_cells = [tuple(x) for x in meta.get("dest_cells", [[cols - 1, j] for j in range(rows)])]
    penalty_cells = [tuple(x) for x in meta.get("penalty_cells", [])]

    mdp, _, _, _ = generate_warehouse_path_concentration_mdp(
        rows=rows,
        cols=cols,
        initial_dist=meta.get("initial_dist"),
        dest_cells=dest_cells,
        is_stochastic=bool(meta.get("stochastic", False)),
        p_correct=float(meta.get("p_correct", 0.8)),
    )

    animate_grid_occupancy(
        mdp=mdp,
        policy=policy,
        horizon=horizon,
        interval_ms=args.interval_ms,
        dest_cells=dest_cells,
        penalty_cells=penalty_cells,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
