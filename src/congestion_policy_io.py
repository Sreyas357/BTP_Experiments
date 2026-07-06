"""Load/save policies for congestion experiments.

Policies from `FusionMDPSolver.solve()` are JSON-serializable as:
- list[timestep] of dict[state_name] -> dict[action_name] -> prob

We store:
- metadata (grid, stochastic params, destination cells, etc.)
- the policy itself

This module is intentionally small to avoid duplicating IO logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def save_policy_json(path: str | Path, *, metadata: dict[str, Any], policy: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata,
        "policy": policy,
    }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_policy_json(path: str | Path) -> dict[str, Any]:
    in_path = Path(path)
    payload = json.loads(in_path.read_text())
    if not isinstance(payload, dict) or "policy" not in payload:
        raise ValueError("Invalid policy file: expected a JSON object with key 'policy'")
    return payload


def metadata_from_spec(spec: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    if is_dataclass(spec):
        base = asdict(spec)
    elif isinstance(spec, dict):
        base = dict(spec)
    else:
        # Best-effort
        base = {k: getattr(spec, k) for k in dir(spec) if not k.startswith("_")}

    if extra:
        base.update(extra)
    return base
