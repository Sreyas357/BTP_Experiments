from mdp import MDP
from mdp.reward import LinearReward, MaxLinearReward
import numpy as np


class CongestionExample:
    def __init__(
        self,
        rows: int,
        cols: int,
        dest_cells=None,
        initial_dist=None,
        capacity: float = 0.2,
        move_cost: float = 1.0,
        congestion_penalty: float = 5.0,
        stochastic: bool = False,
        p_correct: float = 0.8,
    ):
        self.rows = rows
        self.cols = cols
        self.dest_cells = dest_cells
        self.initial_dist = initial_dist
        self.capacity = capacity
        self.move_cost = move_cost
        self.congestion_penalty = congestion_penalty
        self.stochastic = stochastic
        self.p_correct = p_correct

        self.mdp = None
        self.dest_states = []

    def _state(self, i: int, j: int) -> str:
        return f"s_{i}_{j}"

    def generate_congestion_mdp(self):
        actions = ["up", "down", "left", "right", "stay"]

        states = [self._state(i, j) for i in range(self.cols) for j in range(self.rows)]
        state_idx = {s: i for i, s in enumerate(states)}

        if self.dest_cells is None:
            self.dest_cells = [(self.cols - 1, j) for j in range(self.rows)]

        self.dest_states = [self._state(i, j) for (i, j) in self.dest_cells]
        dest_idx = np.array([state_idx[s] for s in self.dest_states], dtype=np.int32)
        if len(dest_idx) == 0:
            raise ValueError("dest_cells must contain at least one destination state")

        state_action_map = {}
        for i in range(self.cols):
            for j in range(self.rows):
                s = self._state(i, j)
                acts = ["stay"]
                if j < self.rows - 1:
                    acts.append("up")
                if j > 0:
                    acts.append("down")
                if i > 0:
                    acts.append("left")
                if i < self.cols - 1:
                    acts.append("right")
                state_action_map[s] = acts

        def get_neighbors(i, j):
            neighbors = [self._state(i, j)]
            if j < self.rows - 1:
                neighbors.append(self._state(i, j + 1))
            if j > 0:
                neighbors.append(self._state(i, j - 1))
            if i > 0:
                neighbors.append(self._state(i - 1, j))
            if i < self.cols - 1:
                neighbors.append(self._state(i + 1, j))
            return neighbors

        transition_prob = {}
        for i in range(self.cols):
            for j in range(self.rows):
                s = self._state(i, j)
                neighbors = get_neighbors(i, j)

                for a in state_action_map[s]:
                    if a == "stay":
                        intended = s
                    elif a == "up":
                        intended = self._state(i, j + 1)
                    elif a == "down":
                        intended = self._state(i, j - 1)
                    elif a == "left":
                        intended = self._state(i - 1, j)
                    else:
                        intended = self._state(i + 1, j)

                    if not self.stochastic:
                        transition_prob[(s, a, intended)] = 1.0
                    else:
                        n = len(neighbors)
                        for ns in neighbors:
                            if ns == intended:
                                prob = self.p_correct
                            else:
                                prob = (1 - self.p_correct) / (n - 1)
                            transition_prob[(s, a, ns)] = prob

        if self.initial_dist is None:
            initial_dist = {s: 0.0 for s in states}
            for j in range(self.rows):
                initial_dist[self._state(0, j)] = 1.0 / self.rows
        else:
            initial_dist = {s: 0.0 for s in states}
            for s, p in self.initial_dist.items():
                initial_dist[s] = p

        reward_model = []

        # move_cost * min_{dest} mu(dest) = -move_cost * max_{dest}(-mu(dest))
        dest_vectors = []
        for i in dest_idx:
            v = np.zeros(len(states))
            v[i] = -1.0
            dest_vectors.append(v)
        reward_model.append(
            MaxLinearReward(
                vectors=dest_vectors,
                offsets=[0.0] * len(dest_vectors),
                weight=-self.move_cost,
            )
        )

        for i in range(len(states)):
            v = np.zeros(len(states))
            v[i] = 1.0
            reward_model.append(
                MaxLinearReward(
                    vectors=[v, np.zeros(len(states))],
                    offsets=[-self.capacity, 0.0],
                    weight=-self.congestion_penalty,
                )
            )

        self.mdp = MDP(
            states=states,
            actions=actions,
            state_action_map=state_action_map,
            transition_prob=transition_prob,
            initial_dist=initial_dist,
            reward_function=None,
            reward_model=reward_model,
            is_concave_reward=True,
        )
        return self.mdp

    def validate_policy(self, policy, gamma: float = 1.0):
        if self.mdp is None:
            raise ValueError("Call generate_congestion_mdp() before validate_policy().")

        mdp = self.mdp
        states = mdp.states
        state_idx = {s: i for i, s in enumerate(states)}
        S = len(states)
        T = len(policy)

        mu = np.zeros((T + 1, S), dtype=float)
        for s, p in mdp.initial_dist.items():
            mu[0, state_idx[s]] = float(p)

        for t in range(T):
            for s in states:
                i = state_idx[s]
                if mu[t, i] <= 0.0:
                    continue
                for a, pi_sa in policy[t].get(s, {}).items():
                    if a not in mdp.state_action_map[s] or pi_sa <= 0.0:
                        continue
                    flow = mu[t, i] * float(pi_sa)
                    if flow <= 0.0:
                        continue
                    for s_next in states:
                        p = mdp.transition_prob.get((s, a, s_next), 0.0)
                        if p > 0.0:
                            mu[t + 1, state_idx[s_next]] += flow * p

        dest_set = set(self.dest_states)

        # Cumulative reached mass by time t (distributional semantics).
        # We track active mass that has not reached destination yet, then
        # accumulate first-hit mass entering the destination set.
        active = np.zeros((T + 1, S), dtype=float)
        for s, p in mdp.initial_dist.items():
            if s not in dest_set:
                active[0, state_idx[s]] = float(p)
        reached = np.zeros(T + 1, dtype=float)

        for t in range(T):
            for s in states:
                i = state_idx[s]
                mass = active[t, i]
                if mass <= 0.0:
                    continue

                for a, pi_sa in policy[t].get(s, {}).items():
                    if a not in mdp.state_action_map[s] or pi_sa <= 0.0:
                        continue
                    flow = mass * float(pi_sa)
                    if flow <= 0.0:
                        continue

                    for s_next in states:
                        p = mdp.transition_prob.get((s, a, s_next), 0.0)
                        if p <= 0.0:
                            continue
                        contrib = flow * p
                        if s_next in dest_set:
                            reached[t + 1] += contrib
                        else:
                            active[t + 1, state_idx[s_next]] += contrib

        destination_progress = np.cumsum(reached)
        destination_progress_percent = [100.0 * float(v) for v in destination_progress]

        congestion_count_per_cell = {}
        for s in states:
            i = state_idx[s]
            congestion_count_per_cell[s] = int(np.sum(mu[:, i] > self.capacity + 1e-12))

        total_reward = 0.0
        for t in range(T):
            r_t = 0.0
            for term in mdp.reward_model:
                if isinstance(term, LinearReward):
                    idx = np.asarray(term.indices, dtype=np.int32)
                    r_t += float(np.dot(np.asarray(term.weights), mu[t, idx]) + term.offset)
                else:
                    vals = [float(np.dot(v, mu[t]) + b) for v, b in zip(term.vectors, term.offsets)]
                    r_t += float(term.weight * max(vals))
            total_reward += (gamma ** t) * r_t

        return {
            "congestion_count_per_cell": congestion_count_per_cell,
            "destination_progress": [float(v) for v in destination_progress],
            "destination_progress_percent": destination_progress_percent,
            "final_mu": {s: float(mu[T, state_idx[s]]) for s in states},
            "total_reward": float(total_reward),
        }


def generate_congestion_mdp(
    rows: int,
    cols: int,
    dest_cells=None,
    initial_dist=None,
    capacity: float = 0.2,
    move_cost: float = 1.0,
    congestion_penalty: float = 5.0,
    stochastic: bool = False,
    p_correct: float = 0.8
):
    return CongestionExample(
        rows=rows,
        cols=cols,
        dest_cells=dest_cells,
        initial_dist=initial_dist,
        capacity=capacity,
        move_cost=move_cost,
        congestion_penalty=congestion_penalty,
        stochastic=stochastic,
        p_correct=p_correct,
    ).generate_congestion_mdp()