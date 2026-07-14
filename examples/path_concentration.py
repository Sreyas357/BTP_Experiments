from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mdp import MDP
from mdp.reward import LinearReward, MaxLinearReward


class Warehouse_path_concentration:
    def __init__(
        self,
        rows: int = 2,
        cols: int = 5,
        time_penalty: float = 1.0,
        concentration_penalty: float = 2.0,
        penalty_states: Optional[Sequence[str]] = None,
        penalty_states_cost: float = 5.0,
        num_random_penalty_states: Optional[int] = None,
        initial_dist: Optional[Dict[str, float]] = None,
        dest_cells: Optional[Sequence[Tuple[int, int]]] = None,
        is_stochastic: bool = False,
        p_correct: float = 0.8,
        random_seed: Optional[int] = None,
    ):
        if rows < 1 or cols < 2:
            raise ValueError("rows must be >= 1 and cols must be >= 2")
        if is_stochastic and not (0.0 < p_correct < 1.0):
            raise ValueError("p_correct must be in (0, 1) for stochastic dynamics")

        self.rows = rows
        self.cols = cols
        self.time_penalty = time_penalty
        self.concentration_penalty = concentration_penalty
        self.penalty_states = list(penalty_states) if penalty_states is not None else None
        self.penalty_states_cost = penalty_states_cost
        self.num_random_penalty_states = num_random_penalty_states
        self.initial_dist = initial_dist
        self.dest_cells = list(dest_cells) if dest_cells is not None else None
        self.is_stochastic = is_stochastic
        self.p_correct = p_correct
        self.random_seed = random_seed

        self.selected_penalty_states: List[str] = []
        self.destination_states: List[str] = []

    def _state(self, i: int, j: int) -> str:
        return f"s_{i}_{j}"

    def _build_state_action_map(self) -> Dict[str, List[str]]:
        state_action_map: Dict[str, List[str]] = {}
        for i in range(self.cols):
            for j in range(self.rows):
                s = self._state(i, j)
                acts = ["stay"]
                if j < self.rows - 1:
                    acts.append("up")
                if j > 0:
                    acts.append("down")
                if i < self.cols - 1:
                    acts.append("right")
                state_action_map[s] = acts
        return state_action_map

    def _build_transition_prob(
        self, state_action_map: Dict[str, List[str]]
    ) -> Dict[Tuple[str, str, str], float]:
        transition_prob: Dict[Tuple[str, str, str], float] = {}

        def neighbors(i: int, j: int) -> List[str]:
            ns = [self._state(i, j)]
            if j < self.rows - 1:
                ns.append(self._state(i, j + 1))
            if j > 0:
                ns.append(self._state(i, j - 1))
            if i < self.cols - 1:
                ns.append(self._state(i + 1, j))
            return ns

        for i in range(self.cols):
            for j in range(self.rows):
                s = self._state(i, j)
                local_neighbors = neighbors(i, j)

                for a in state_action_map[s]:
                    if a == "stay":
                        intended = s
                    elif a == "up":
                        intended = self._state(i, j + 1)
                    elif a == "down":
                        intended = self._state(i, j - 1)
                    else:
                        intended = self._state(i + 1, j)

                    if not self.is_stochastic:
                        transition_prob[(s, a, intended)] = 1.0
                    else:
                        n = len(local_neighbors)
                        for ns in local_neighbors:
                            if ns == intended:
                                prob = self.p_correct
                            else:
                                prob = (1.0 - self.p_correct) / (n - 1)
                            transition_prob[(s, a, ns)] = prob

        return transition_prob

    def _build_initial_dist(self, states: List[str]) -> Dict[str, float]:
        if self.initial_dist is None:
            initial_dist = {s: 0.0 for s in states}
            for j in range(self.rows):
                initial_dist[self._state(0, j)] = 1.0 / self.rows
            return initial_dist

        initial_dist = {s: 0.0 for s in states}
        for s, p in self.initial_dist.items():
            initial_dist[s] = p
        return initial_dist

    def _resolve_destination_states(self) -> List[str]:
        if self.dest_cells is None:
            cells = [(self.cols - 1, j) for j in range(self.rows)]
        else:
            cells = list(self.dest_cells)
        return [self._state(i, j) for (i, j) in cells]

    def _resolve_penalty_states(self, states: List[str], destination_states: List[str]) -> List[str]:
        if self.penalty_states is not None:
            return list(self.penalty_states)

        candidates = [s for s in states if s not in set(destination_states)]
        if not candidates:
            return []

        if self.num_random_penalty_states is None:
            k = max(1, min(self.rows, len(candidates) // 3 if len(candidates) >= 3 else 1))
        else:
            k = max(0, min(int(self.num_random_penalty_states), len(candidates)))

        rng = np.random.default_rng(self.random_seed)
        picked = rng.choice(candidates, size=k, replace=False)
        return [str(x) for x in picked.tolist()]

    def generate_mdp(self):
        states = [self._state(i, j) for i in range(self.cols) for j in range(self.rows)]
        state_idx = {s: i for i, s in enumerate(states)}

        actions = ["right", "up", "down", "stay"]
        state_action_map = self._build_state_action_map()
        transition_prob = self._build_transition_prob(state_action_map)
        initial_dist = self._build_initial_dist(states)

        self.destination_states = self._resolve_destination_states()
        destination_set = set(self.destination_states)
        non_destination_indices = [i for i, s in enumerate(states) if s not in destination_set]

        self.selected_penalty_states = self._resolve_penalty_states(states, self.destination_states)
        penalty_indices = [state_idx[s] for s in self.selected_penalty_states]

        # Corridor masses by row: phi_j(mu) = sum_{i} mu(s_{i,j}).
        row_vectors: List[np.ndarray] = []
        for j in range(self.rows):
            v = np.zeros(len(states), dtype=float)
            for i in range(self.cols):
                v[state_idx[self._state(i, j)]] = 1.0
            row_vectors.append(v)

        reward_model = [
            # Time penalty on mass outside destination.
            LinearReward(
                indices=non_destination_indices,
                weights=[-self.time_penalty] * len(non_destination_indices),
                offset=0.0,
            ),
            # Concentration reward: +lambda * max_j phi_j(mu) - lambda.
            LinearReward(indices=[], weights=[], offset=-self.concentration_penalty),
            MaxLinearReward(
                vectors=row_vectors,
                offsets=[0.0] * len(row_vectors),
                weight=self.concentration_penalty,
            ),
            # Penalty-state cost.
            LinearReward(
                indices=penalty_indices,
                weights=[-self.penalty_states_cost] * len(penalty_indices),
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

        row0 = [self._state(i, 0) for i in range(self.cols)]
        row1 = [self._state(i, 1) for i in range(self.cols)] if self.rows > 1 else []

        return mdp, row0, row1, list(self.destination_states)


def generate_warehouse_path_concentration_mdp(
    rows: int = 2,
    cols: int = 5,
    time_penalty: float = 1.0,
    concentration_penalty: float = 2.0,
    penalty_states: Optional[Sequence[str]] = None,
    penalty_states_cost: float = 5.0,
    num_random_penalty_states: Optional[int] = None,
    initial_dist: Optional[Dict[str, float]] = None,
    dest_cells: Optional[Sequence[Tuple[int, int]]] = None,
    is_stochastic: bool = False,
    p_correct: float = 0.8,
    random_seed: Optional[int] = None,
):
    return Warehouse_path_concentration(
        rows=rows,
        cols=cols,
        time_penalty=time_penalty,
        concentration_penalty=concentration_penalty,
        penalty_states=penalty_states,
        penalty_states_cost=penalty_states_cost,
        num_random_penalty_states=num_random_penalty_states,
        initial_dist=initial_dist,
        dest_cells=dest_cells,
        is_stochastic=is_stochastic,
        p_correct=p_correct,
        random_seed=random_seed,
    ).generate_mdp()