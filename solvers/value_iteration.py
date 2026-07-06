from itertools import product
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from mdp.reward import LinearReward, MaxLinearReward


class MaxLinearValueIterationSolver:
    """
    Backward value iteration for max-linear reward MDPs.

    Value function representation (folded alpha vectors)
    ----------------------------------------------------
    Each alpha vector is stored as  v̄ = v + c·1  (where 1 is the all-ones
    vector), so that for any μ ∈ Δ(S):

        v̄ · μ  =  v · μ + c · (Σ_s μ_s)  =  v · μ + c

    i.e. the constant offset folds away because distributions sum to 1.
    gamma_sets[t] is a 2-D numpy array of shape [K_t, S].

    Dominance over Δ(S)
    -------------------
    v̄_j dominates v̄_i  iff  v̄_j · μ ≥ v̄_i · μ  ∀μ ∈ Δ(S).
    Since minimising a linear function over the simplex picks the smallest
    component, this reduces to the pointwise condition  v̄_j[s] ≥ v̄_i[s] ∀s,
    which is both necessary and sufficient — no LP needed.

    Bellman backup
    --------------
    V_t(μ) = R(μ) + γ · max_π V_{t+1}(T_π^T μ)

    With max-linear reward  R(μ) = max_m [ā_m · μ]  and
    V_{t+1}(μ') = max_k [v̄_k · μ'],

        V_t(μ) = max_{m,π,k} [ (ā_m + γ · T_π v̄_k) · μ ]

    Each candidate alpha vector is  ā_m + γ · T_π v̄_k.

    Performance: incremental Pareto update
    ---------------------------------------
    Policies are processed in batches.  After each batch the partial Pareto
    front is updated, bounding its size by the true Pareto front size at all
    times.  This avoids accumulating O(M·P·K) candidates before the first prune
    (which causes O((MPK)²·S) work with a bulk approach).

    PBVI-style approximate pruning
    --------------------------------
    When max_vectors is set, Point-Based pruning replaces exact dominance
    pruning.  A set of belief points B ⊆ Δ(S) is used: for each b ∈ B the
    alpha vector that maximises b · v̄ is kept.  This bounds |Γ_t| ≤ max_vectors.
    The approximation discards vectors that are optimal nowhere in B.
    Increasing n_belief_points tightens the approximation.
    """

    def __init__(
        self,
        mdp,
        horizon: int,
        gamma: float,
        prune: bool = True,
        prune_tol: float = 1e-12,
        # PBVI approximate pruning
        max_vectors: Optional[int] = None,
        n_belief_points: int = 500,
        belief_points: Optional[np.ndarray] = None,
        seed: int = 42,
        # Batch size for incremental Pareto updates in solve()
        policy_batch_size: int = 64,
        # Cache full policy/T_stack only when policy count is manageable.
        max_cached_policies: int = 200000,
    ):
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        if mdp.reward_model is None:
            raise ValueError("mdp.reward_model is required")
        if max_vectors is not None and max_vectors < 1:
            raise ValueError("max_vectors must be >= 1")

        self.mdp = mdp
        self.horizon = horizon
        self.gamma = gamma
        self.prune = prune
        self.prune_tol = prune_tol
        self.max_vectors = max_vectors
        self.policy_batch_size = policy_batch_size
        self.max_cached_policies = max_cached_policies

        self.states = list(mdp.states)
        self.state_idx = {s: i for i, s in enumerate(self.states)}
        self.S = len(self.states)
        self._action_lists = [self.mdp.state_action_map[s] for s in self.states]
        self.policy_count = self._count_policies()

        # Cache policies and transitions only for smaller policy spaces.
        self._use_cached_policies = self.policy_count <= self.max_cached_policies
        if self._use_cached_policies:
            self.policies = self._enumerate_deterministic_policies()
            self.T_stack = np.array(
                [self._build_policy_transition(p) for p in self.policies],
                dtype=float,
            )
        else:
            self.policies = None
            self.T_stack = None

        # Folded reward modes  ā_m = a_m + b_m·1;  shape [M, S]
        self._R_bar: np.ndarray = self._build_reward_modes_folded()

        # Belief points for PBVI
        if max_vectors is not None:
            if belief_points is not None:
                bp = np.asarray(belief_points, dtype=float)
                if bp.ndim != 2 or bp.shape[1] != self.S:
                    raise ValueError(
                        f"belief_points must have shape [B, {self.S}], got {bp.shape}"
                    )
                self.belief_points: Optional[np.ndarray] = bp
            else:
                rng = np.random.default_rng(seed)
                self.belief_points = rng.dirichlet(
                    np.ones(self.S), size=n_belief_points
                )
        else:
            self.belief_points = None

        # ---- legacy attributes kept for backward compatibility ----
        self.transition_matrices: List[np.ndarray] = (
            list(self.T_stack) if self.T_stack is not None else []
        )
        self.reward_modes: List[Tuple[np.ndarray, float]] = self._build_reward_modes()
        # -----------------------------------------------------------

        # Filled by solve(); each entry is a [K_t, S] array.
        self.gamma_sets: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> List[np.ndarray]:
        """
        Run backward value iteration.

        Returns
        -------
        gamma_sets : list of length horizon+1
            gamma_sets[t] is a numpy array of shape [K_t, S] containing the
            K_t folded alpha vectors at time t.
        """
        S, gamma = self.S, self.gamma
        R_bar = self._R_bar      # [M, S]
        B = self.policy_batch_size

        gamma_sets: List[Optional[np.ndarray]] = [None] * (self.horizon + 1)
        # Terminal: V_H(μ) = 0  →  single zero alpha vector
        gamma_sets[self.horizon] = np.zeros((1, S), dtype=float)

        for t in range(self.horizon - 1, -1, -1):
            G_next = gamma_sets[t + 1]  # [K, S]

            # Build Pareto front incrementally over policy batches.
            # Processing all P × M × K candidates at once before pruning creates
            # an O((P·M·K)²·S) pruning step.  Batching keeps the intermediate
            # candidate count bounded by B·M·K + |current Pareto front|.
            pareto_front = np.empty((0, S), dtype=float)

            for _, T_batch in self._iter_policy_batches(B):
                # batch_future[b, k, s] = γ · Σ_{s'} T_batch[b, s, s'] · G_next[k, s']
                batch_future = gamma * (T_batch @ G_next.T).transpose(0, 2, 1)  # [b,K,S]
                # Expand reward modes: [M, b, K, S] → [M·b·K, S]
                batch_cands = (
                    R_bar[:, np.newaxis, np.newaxis, :] + batch_future[np.newaxis]
                ).reshape(-1, S)

                pareto_front = self._merge_into_pareto(batch_cands, pareto_front)

            if self.max_vectors is not None:
                pareto_front = self._pbvi_prune(pareto_front)

            gamma_sets[t] = pareto_front

        self.gamma_sets = gamma_sets
        return gamma_sets

    def value(self, mu: Dict[str, float], t: int = 0) -> float:
        """Optimal value of distribution mu at time t."""
        if not self.gamma_sets:
            raise ValueError("Call solve() first")
        if not (0 <= t <= self.horizon):
            raise ValueError("t must be in [0, horizon]")
        mu_vec = self._mu_to_vec(mu)
        return float(np.max(self.gamma_sets[t] @ mu_vec))

    def best_policy_for_distribution(
        self, mu: Dict[str, float], t: int
    ) -> Dict[str, str]:
        """Return the deterministic policy that achieves V_t(mu)."""
        if not self.gamma_sets:
            raise ValueError("Call solve() first")
        if not (0 <= t < self.horizon):
            raise ValueError("t must be in [0, horizon-1]")
        mu_vec = self._mu_to_vec(mu)
        best_policy, _ = self._best_policy_and_transition(mu_vec, t)
        return best_policy

    def rollout_distribution(
        self, mu0: Dict[str, float]
    ) -> List[Dict[str, float]]:
        """
        Follow the greedy policy from mu0, return distributions
        μ_0, μ_1, ..., μ_horizon.
        """
        if not self.gamma_sets:
            raise ValueError("Call solve() first")
        mu_traj = [dict(mu0)]
        current = self._mu_to_vec(mu0)
        for t in range(self.horizon):
            _, T_pi = self._best_policy_and_transition(current, t)
            # μ_{t+1}[s'] = Σ_s T_π[s,s'] μ_t[s]
            current = T_pi.T @ current
            mu_traj.append(self._vec_to_mu(current))
        return mu_traj

    # ------------------------------------------------------------------
    # Internal: policy selection
    # ------------------------------------------------------------------

    def _best_policy_and_transition(
        self, mu_vec: np.ndarray, t: int
    ) -> Tuple[Dict[str, str], np.ndarray]:
        """
        Policy and transition matrix that achieve V_t(mu_vec).

        Key decomposition: since the current reward R(μ) = max_m[ā_m · μ] is
        independent of the chosen policy π, the best policy is the one that
        maximises the NEXT-STEP value:

            p* = argmax_p  V_{t+1}(T_p^T μ)
               = argmax_p  max_k [v̄_k · (T_p^T μ)]

        This avoids building an [M, P, K, S] intermediate array.
        """
        G_next = self.gamma_sets[t + 1]  # [K, S]

        if self._use_cached_policies:
            pushed_mus = np.einsum("pij,i->pj", self.T_stack, mu_vec)  # [P, S]
            best_future = self.gamma * (G_next @ pushed_mus.T).max(axis=0)  # [P]
            pi_idx = int(np.argmax(best_future))
            return dict(self.policies[pi_idx]), self.T_stack[pi_idx]

        best_score = -np.inf
        best_policy: Optional[Dict[str, str]] = None
        best_T: Optional[np.ndarray] = None
        for pol_batch, T_batch in self._iter_policy_batches(self.policy_batch_size):
            pushed_mus = np.einsum("bij,i->bj", T_batch, mu_vec)  # [b, S]
            scores = self.gamma * (G_next @ pushed_mus.T).max(axis=0)  # [b]
            local_idx = int(np.argmax(scores))
            local_score = float(scores[local_idx])
            if local_score > best_score:
                best_score = local_score
                best_policy = dict(pol_batch[local_idx])
                best_T = T_batch[local_idx]

        if best_policy is None or best_T is None:
            raise RuntimeError("No deterministic policies were generated")
        return best_policy, best_T

    # ------------------------------------------------------------------
    # Internal: Pareto-front maintenance
    # ------------------------------------------------------------------

    def _merge_into_pareto(
        self, new_cands: np.ndarray, front: np.ndarray
    ) -> np.ndarray:
        """
        Merge new_cands into the current Pareto front (front).

        Exploits the fact that front is already a valid Pareto front to avoid
        the O(N²) pairwise check on the combined set:

        1. Filter new_cands dominated by front:   O(|front| · |new| · S)
        2. Remove front vectors dominated by any surviving new candidate:
                                                   O(|survivors| · |front| · S)
        3. Prune within the surviving new candidates (they were not pre-filtered
           against each other):                    O(|survivors|² · S)
        4. Concatenate.

        Because step 1 typically reduces |survivors| to a small number (bounded
        by the true Pareto front size minus the current front size), steps 2–4
        are cheap in practice.
        """
        if len(new_cands) == 0:
            return front
        if len(front) == 0:
            return self._prune_dominated(new_cands)

        tol = self.prune_tol

        # ── Step 1: remove new_cands weakly dominated by front ─────────
        # diff[j, c, s] = front[j, s] - new_cands[c, s]
        # c is redundant if any f satisfies f[s] >= c[s] - tol for ALL s
        # (covers both strict dominance and equality / near-duplicates).
        diff_jc = front[:, np.newaxis, :] - new_cands[np.newaxis, :, :]  # [J, C, S]
        j_weakdom_c = np.all(diff_jc >= -tol, axis=2)                    # [J, C]
        survivors = new_cands[~j_weakdom_c.any(axis=0)]

        if len(survivors) == 0:
            return front

        # ── Step 2: remove front vectors weakly dominated by survivors ──
        diff_fj = survivors[:, np.newaxis, :] - front[np.newaxis, :, :]  # [C', J, S]
        s_weakdom_f = np.all(diff_fj >= -tol, axis=2)                    # [C', J]
        front = front[~s_weakdom_f.any(axis=0)]

        # ── Step 3: prune within survivors ──────────────────────────────
        if len(survivors) > 1:
            survivors = self._prune_dominated(survivors)

        if len(front) == 0:
            return survivors
        return np.vstack([front, survivors])

    def _prune_dominated(self, cands: np.ndarray) -> np.ndarray:
        """
        Exact dominance pruning on folded alpha vectors.

        cands : [N, S]
        Removes v̄_i whenever ∃j ≠ i s.t. v̄_j[s] ≥ v̄_i[s] ∀s (strict somewhere).
        This is equivalent to dominance over all μ ∈ Δ(S) (see class docstring).
        """
        if len(cands) <= 1:
            return cands

        # Deduplicate first (eliminates trivially redundant candidates cheaply)
        cands = np.unique(np.round(cands, 12), axis=0)
        N = len(cands)
        if N <= 1:
            return cands

        tol = self.prune_tol
        keep = np.ones(N, dtype=bool)

        for i in range(N):
            if not keep[i]:
                continue
            # diff[j, s] = cands[j, s] - cands[i, s]
            diff = cands - cands[i]          # [N, S]  (broadcast row i)
            # j dominates i: diff[j] ≥ -tol everywhere AND diff[j] ≥ tol somewhere
            dominated_by = (
                np.all(diff >= -tol, axis=1) & np.any(diff >= tol, axis=1)
            )
            dominated_by[i] = False
            if dominated_by.any():
                keep[i] = False

        return cands[keep]

    def _pbvi_prune(self, cands: np.ndarray) -> np.ndarray:
        """
        Point-Based Value Iteration (PBVI-style) approximate pruning.

        Algorithm
        ---------
        1. Score every candidate against every belief point:
               scores[n, b] = cands[n] · B[b]
        2. For each b, record the index of the highest-scoring candidate.
        3. Keep only the unique winners.  This bounds |Γ_t| ≤ |B|.
        4. If the number of unique winners exceeds max_vectors, retain the
           max_vectors vectors with the highest coverage (= number of belief
           points for which they are the unique winner).
        """
        B = self.belief_points           # [n_b, S]
        scores = cands @ B.T             # [N, n_b]

        best_idx = np.argmax(scores, axis=0)           # [n_b]
        unique_idx, counts = np.unique(best_idx, return_counts=True)

        if len(unique_idx) <= self.max_vectors:
            return cands[unique_idx]

        # More winners than budget: keep max_vectors by coverage
        top_k = np.argpartition(counts, -self.max_vectors)[-self.max_vectors :]
        return cands[unique_idx[top_k]]

    # ------------------------------------------------------------------
    # Internal: reward modes
    # ------------------------------------------------------------------

    def _build_reward_modes_folded(self) -> np.ndarray:
        """
        Build folded reward mode vectors  ā_m = a_m + b_m·1,  shape [M, S].

        Since ā_m · μ = a_m · μ + b_m for any μ ∈ Δ(S), the scalar offset
        is absorbed into the vector without changing the value function.

        Backup correctness: the folded representation is closed under the
        backup because T_π is row-stochastic (T_π · 1 = 1), so
            ā_m + γ · T_π v̄_k
            = (a_m + b_m·1) + γ · T_π (v_k + c_k·1)
            = (a_m + γ T_π v_k) + (b_m + γ c_k)·1
        which is the folded version of (v_new, c_new). ✓
        """
        modes = self._build_reward_modes()
        return np.array([a + b for a, b in modes], dtype=float)

    def _build_reward_modes(self) -> List[Tuple[np.ndarray, float]]:
        """
        Enumerate all (a_mode, b_mode) reward-mode pairs.
        LinearReward shifts all modes uniformly; MaxLinearReward multiplies
        the mode count by its number of linear pieces.
        """
        modes: List[Tuple[np.ndarray, float]] = [(np.zeros(self.S, dtype=float), 0.0)]

        for term in self.mdp.reward_model:
            if isinstance(term, LinearReward):
                step = np.zeros(self.S, dtype=float)
                for idx, w in zip(term.indices, term.weights):
                    step[int(idx)] += float(w)
                modes = [(v + step, c + float(term.offset)) for v, c in modes]

            elif isinstance(term, MaxLinearReward):
                if term.weight < 0:
                    raise ValueError(
                        "MaxLinearReward.weight must be nonnegative for VI"
                    )
                expanded: List[Tuple[np.ndarray, float]] = []
                for v, c in modes:
                    for vec, off in zip(term.vectors, term.offsets):
                        expanded.append((
                            v + float(term.weight) * np.asarray(vec, dtype=float),
                            c + float(term.weight) * float(off),
                        ))
                modes = expanded
            else:
                raise ValueError(f"Unsupported reward term type: {type(term)}")

        return modes

    # ------------------------------------------------------------------
    # Internal: MDP helpers
    # ------------------------------------------------------------------

    def _count_policies(self) -> int:
        total = 1
        for acts in self._action_lists:
            total *= len(acts)
        return total

    def _iter_policy_batches(self, batch_size: int):
        """
        Yield batches of deterministic policies and their transition matrices.

        Returns
        -------
        (policies_batch, T_batch)
            policies_batch: list[dict[str, str]]
            T_batch: ndarray [b, S, S]
        """
        if self._use_cached_policies:
            for start in range(0, len(self.policies), batch_size):
                pol_batch = self.policies[start : start + batch_size]
                yield pol_batch, self.T_stack[start : start + batch_size]
            return

        pol_batch: List[Dict[str, str]] = []
        T_batch: List[np.ndarray] = []
        for chosen in product(*self._action_lists):
            policy = {s: a for s, a in zip(self.states, chosen)}
            pol_batch.append(policy)
            T_batch.append(self._build_policy_transition(policy))
            if len(pol_batch) == batch_size:
                yield pol_batch, np.asarray(T_batch, dtype=float)
                pol_batch = []
                T_batch = []

        if pol_batch:
            yield pol_batch, np.asarray(T_batch, dtype=float)

    def _enumerate_deterministic_policies(self) -> List[Dict[str, str]]:
        return [
            {s: a for s, a in zip(self.states, chosen)}
            for chosen in product(*self._action_lists)
        ]

    def _build_policy_transition(self, policy: Dict[str, str]) -> np.ndarray:
        T_pi = np.zeros((self.S, self.S), dtype=float)
        for s in self.states:
            a = policy[s]
            i = self.state_idx[s]
            for s_next in self.states:
                T_pi[i, self.state_idx[s_next]] = self.mdp.transition_prob[(s, a, s_next)]
        return T_pi

    def _mu_to_vec(self, mu: Dict[str, float]) -> np.ndarray:
        vec = np.zeros(self.S, dtype=float)
        for s in self.states:
            vec[self.state_idx[s]] = float(mu[s])
        return vec

    def _vec_to_mu(self, vec: np.ndarray) -> Dict[str, float]:
        return {s: float(vec[self.state_idx[s]]) for s in self.states}
