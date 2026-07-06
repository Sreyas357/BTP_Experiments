import mosek.fusion as mf
import numpy as np
from pydantic import BaseModel, field_validator
from typing import Any
from mdp.reward import *


# ============================================================
# VALIDATION
# ============================================================

class SolverConfig(BaseModel):
    mdp: Any
    horizon: int
    gamma: float
    precision: float | None = None

    @field_validator("horizon")
    @classmethod
    def check_horizon(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("horizon must be positive int")
        return v

    @field_validator("gamma")
    @classmethod
    def check_gamma(cls, v):
        if not (0 < v <= 1):
            raise ValueError("gamma must be in (0,1]")
        return v

    @field_validator("mdp")
    @classmethod
    def check_mdp(cls, v):
        required = [
            "states",
            "actions",
            "transition_prob",
            "initial_dist",
            "reward_model",
            "is_concave_reward"
        ]
        for attr in required:
            if not hasattr(v, attr):
                raise ValueError(f"mdp missing attribute: {attr}")
        return v

    @field_validator("precision")
    @classmethod
    def check_precision(cls, v):
        if v is None:
            return v
        if v <= 0:
            raise ValueError("precision must be > 0 when provided")
        return v


# ============================================================
# SOLVER
# ============================================================

class FusionMDPSolver:
    def __init__(
        self,
        mdp,
        horizon: int,
        gamma: float,
        mosek_num_threads: int | None = None,
        precision: float | None = None,
    ):

        config = SolverConfig(mdp=mdp, horizon=horizon, gamma=gamma, precision=precision)

        self.mdp = config.mdp
        self.T = config.horizon
        self.gamma = config.gamma
        self.precision = config.precision
        self.mosek_num_threads = mosek_num_threads

        if self.mosek_num_threads is not None and self.mosek_num_threads <= 0:
            raise ValueError("mosek_num_threads must be positive when provided")

        if not self.mdp.is_concave_reward:
            raise ValueError("Requires concave reward")

        self.states = self.mdp.states
        self.actions = self.mdp.actions

        self.S = len(self.states)
        self.A = len(self.actions)

        self.state_idx = {s: i for i, s in enumerate(self.states)}
        self.action_idx = {a: i for i, a in enumerate(self.actions)}

        self._build_transition_tensor()

    # --------------------------------------------------------
    # TRANSITION TENSOR
    # --------------------------------------------------------
    def _build_transition_tensor(self):
        self.T_mat = np.zeros((self.S, self.A, self.S))

        for (s, a, s_next), p in self.mdp.transition_prob.items():
            i = self.state_idx[s]
            j = self.action_idx[a]
            k = self.state_idx[s_next]
            self.T_mat[i, j, k] = p

    # --------------------------------------------------------
    # SOLVE
    # --------------------------------------------------------
    def solve(self):

        S, A, T = self.S, self.A, self.T

        with mf.Model("mdp") as M:
            if self.mosek_num_threads is not None:
                M.setSolverParam("numThreads", int(self.mosek_num_threads))

            if self.precision is not None:
                # Larger tolerance usually yields faster solves on large problems
                # at the expense of objective/constraint accuracy.
                tol = float(self.precision)
                M.setSolverParam("intpntTolRelGap", tol)
                M.setSolverParam("intpntTolPfeas", tol)
                M.setSolverParam("intpntTolDfeas", tol)
                M.setSolverParam("intpntCoTolRelGap", tol)
                M.setSolverParam("intpntCoTolPfeas", tol)
                M.setSolverParam("intpntCoTolDfeas", tol)

            # Stochastic instances can be numerically harder, and MOSEK may
            # return a feasible solution while the default/basic pointer is
            # not marked Optimal. Accept feasible solutions explicitly.
            M.acceptedSolutionStatus(mf.AccSolutionStatus.Feasible)

            # -------------------------
            # VARIABLES
            # -------------------------
            x = [
                M.variable(f"x_{t}", [S, A], mf.Domain.greaterThan(0.0))
                for t in range(T)
            ]

            mu = [
                M.variable(f"mu_{t}", S, mf.Domain.unbounded())
                for t in range(T + 1)
            ]

            # -------------------------
            # INITIAL DISTRIBUTION
            # -------------------------
            mu0 = np.array([self.mdp.initial_dist[s] for s in self.states])
            M.constraint(mu[0], mf.Domain.equalsTo(mu0))

            # -------------------------
            # FLOW CONSTRAINT
            # -------------------------
            for t in range(T):
                M.constraint(
                    mf.Expr.sub(mf.Expr.sum(x[t], 1), mu[t]),
                    mf.Domain.equalsTo(0.0)
                )

            # -------------------------
            # INVALID ACTION MASK
            # -------------------------
            for t in range(T):
                for i, s in enumerate(self.states):
                    allowed = set(self.mdp.state_action_map[s])
                    for a, a_name in enumerate(self.actions):
                        if a_name not in allowed:
                            M.constraint(x[t].index(i, a), mf.Domain.equalsTo(0.0))

            # -------------------------
            # DYNAMICS
            # -------------------------
            for t in range(T):
                for j in range(S):
                    next_mu_j = mf.Expr.constTerm(0.0)

                    for i in range(S):
                        for a in range(A):
                            p = self.T_mat[i, a, j]
                            if p > 0:
                                next_mu_j = mf.Expr.add(
                                    next_mu_j,
                                    mf.Expr.mul(p, x[t].index(i, a))
                                )

                    M.constraint(
                        mf.Expr.sub(mu[t + 1].index(j), next_mu_j),
                        mf.Domain.equalsTo(0.0)
                    )

            # -------------------------
            # OBJECTIVE
            # -------------------------
            r = M.variable("r", T, mf.Domain.unbounded())

            for t in range(T):
                reward_expr = self._build_reward(M, mu[t])
                M.constraint(
                    mf.Expr.sub(r.index(t), reward_expr),
                    mf.Domain.equalsTo(0.0)
                )

            discounts = np.array([self.gamma ** t for t in range(T)], dtype=float)
            M.objective(mf.ObjectiveSense.Maximize, mf.Expr.dot(discounts, r))

            # -------------------------
            # SOLVE
            # -------------------------
            M.solve()

            sol_type = self._pick_solution_type(M)
            M.selectedSolution(sol_type)

            sel_primal_status = M.getPrimalSolutionStatus(sol_type)
            if sel_primal_status != mf.SolutionStatus.Optimal:
                print(
                    "[FusionMDPSolver] WARNING: selected solution is not certified Optimal. "
                    f"solution_type={sol_type}, primal_status={sel_primal_status}"
                )

                primal_obj = M.primalObjValue()
                dual_obj = None
                try:
                    dual_obj = M.dualObjValue()
                except Exception:
                    dual_obj = None

                if dual_obj is not None:
                    abs_gap = abs(dual_obj - primal_obj)
                    denom = max(1.0, abs(primal_obj), abs(dual_obj))
                    rel_gap = abs_gap / denom
                    print(
                        "[FusionMDPSolver] Estimated objective gap bound: "
                        f"abs_gap={abs_gap:.6e}, rel_gap={rel_gap:.6e}, "
                        f"primal={primal_obj:.6e}, dual={dual_obj:.6e}"
                    )
                else:
                    print(
                        "[FusionMDPSolver] Objective-gap bound unavailable: "
                        "dual objective is not accessible for this solution type/status."
                    )

            return self._extract_policy(x, mu), M.primalObjValue()

    def _pick_solution_type(self, model):
        candidates = [
            mf.SolutionType.Interior,
            mf.SolutionType.Basic,
            mf.SolutionType.Default,
        ]
        allowed = {mf.SolutionStatus.Optimal, mf.SolutionStatus.Feasible}
        seen = {}

        for st in candidates:
            try:
                s = model.getPrimalSolutionStatus(st)
            except Exception:
                continue
            seen[str(st)] = str(s)
            if s in allowed:
                return st

        raise RuntimeError(
            "No usable MOSEK primal solution available. "
            f"Statuses: {seen}"
        )

    # --------------------------------------------------------
    # BUILD REWARD
    # --------------------------------------------------------
    def _build_reward(self, M, mu_t):
        running = M.variable(mf.Domain.unbounded())
        M.constraint(running, mf.Domain.equalsTo(0.0))

        for term in self.mdp.reward_model:

            # -------------------------
            # LINEAR: a^T mu + b
            # -------------------------
            if isinstance(term, LinearReward):
                indices = np.asarray(term.indices, dtype=np.int32)
                vals = mu_t.pick(indices)
                expr = mf.Expr.dot(term.weights, vals)

                if term.offset != 0.0:
                    expr = mf.Expr.add(expr, term.offset)

            # -------------------------
            # MAX-LINEAR: weight * max_i (a_i^T mu + b_i)
            # -------------------------
            elif isinstance(term, MaxLinearReward):

                if term.weight >= 0:
                    raise ValueError(
                        "MaxLinearReward requires weight < 0 for convex optimization"
                    )

                z = M.variable()  # scalar

                for v, b in zip(term.vectors, term.offsets):
                    M.constraint(
                        mf.Expr.sub(
                            z,
                            mf.Expr.add(mf.Expr.dot(v, mu_t), b)
                        ),
                        mf.Domain.greaterThan(0.0)
                    )

                expr = mf.Expr.mul(term.weight, z)

            else:
                raise ValueError(f"Unknown reward term: {type(term)}")

            nxt = M.variable(mf.Domain.unbounded())
            M.constraint(
                mf.Expr.sub(nxt, mf.Expr.add(running, expr)),
                mf.Domain.equalsTo(0.0)
            )
            running = nxt

        return running

    # --------------------------------------------------------
    # POLICY EXTRACTION
    # --------------------------------------------------------
    def _extract_policy(self, x, mu):

        policy = []

        for t in range(self.T):

            x_val = np.array(x[t].level()).reshape(self.S, self.A)
            mu_val = np.array(mu[t].level())

            pi_t = {}

            for i, s in enumerate(self.states):

                if mu_val[i] > 1e-8:
                    probs = x_val[i] / mu_val[i]
                else:
                    probs = np.ones(self.A) / self.A

                pi_t[s] = {
                    self.actions[a]: probs[a]
                    for a in range(self.A)
                }

            policy.append(pi_t)

        return policy