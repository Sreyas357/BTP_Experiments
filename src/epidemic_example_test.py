import time
import numpy as np
from typing import Dict, List, Tuple

from mdp import *    
from mdp.reward import LinearReward
from solvers.convex_opt import FusionMDPSolver
from solvers.value_iteration import MaxLinearValueIterationSolver

class SEIR_ICU_ControlMDP:
    """
    Generates a Distributional MDP for Epidemic Control using a 5-state SEIR-ICU model.
    Transition probabilities are approximated as discrete-time linear rates based on 
    standard COVID-19 epidemiological parameters (Ferguson et al., 2020).
    """
    def __init__(
        self,
        beta_open: float = 0.20,      # Force of infection (Open)
        beta_lockdown: float = 0.05,  # Force of infection (Lockdown)
        sigma: float = 0.20,          # E -> I rate (~5 days incubation)
        gamma: float = 0.08,          # I -> R rate (~12.5 days recovery)
        alpha: float = 0.02,          # I -> ICU progression rate 
        mu: float = 0.10,             # ICU -> R rate (~10 days ICU stay)
        cost_econ_open: float = 0.0,
        cost_econ_lockdown: float = 5.0,
        cost_treat: float = 10.0,
        cost_critical: float = 50.0,
    ):
        self.disease_states = ["S", "E", "I", "ICU", "R"]
        self.actions = ["Open", "Lockdown"]
        
        self.beta = {"Open": beta_open, "Lockdown": beta_lockdown}
        self.sigma = sigma
        self.gamma = gamma
        self.alpha = alpha
        self.mu = mu
        
        self.cost_econ = {"Open": cost_econ_open, "Lockdown": cost_econ_lockdown}
        self.cost_health = {"S": 0.0, "E": 0.0, "I": cost_treat, "ICU": cost_critical, "R": 0.0}

    def _state(self, s: str, a: str) -> str:
        # Augmenting state space to S' = S x A to standardise action-dependent rewards.
        return f"{s}_{a}"

    def generate_mdp(self) -> MDP:
        states = [self._state(s, a) for s in self.disease_states for a in self.actions]
        state_idx = {s: i for i, s in enumerate(states)}
        state_action_map = {state: self.actions for state in states}
        
        transition_prob = {}
        for s_prev in self.disease_states:
            for a_prev in self.actions:
                curr_state = self._state(s_prev, a_prev)
                
                for a_new in self.actions:
                    # S -> E or S -> S
                    if s_prev == "S":
                        p_inf = self.beta[a_new]
                        transition_prob[(curr_state, a_new, self._state("E", a_new))] = p_inf
                        transition_prob[(curr_state, a_new, self._state("S", a_new))] = 1.0 - p_inf
                    
                    # E -> I or E -> E
                    elif s_prev == "E":
                        transition_prob[(curr_state, a_new, self._state("I", a_new))] = self.sigma
                        transition_prob[(curr_state, a_new, self._state("E", a_new))] = 1.0 - self.sigma

                    # I -> R, I -> ICU, or I -> I
                    elif s_prev == "I":
                        transition_prob[(curr_state, a_new, self._state("R", a_new))] = self.gamma
                        transition_prob[(curr_state, a_new, self._state("ICU", a_new))] = self.alpha
                        transition_prob[(curr_state, a_new, self._state("I", a_new))] = 1.0 - self.gamma - self.alpha
                    
                    # ICU -> R or ICU -> ICU
                    elif s_prev == "ICU":
                        transition_prob[(curr_state, a_new, self._state("R", a_new))] = self.mu
                        transition_prob[(curr_state, a_new, self._state("ICU", a_new))] = 1.0 - self.mu

                    # R -> R (Absorbing)
                    elif s_prev == "R":
                        transition_prob[(curr_state, a_new, self._state("R", a_new))] = 1.0
        
        # Initial distribution (90% S, 10% E, arbitrary previous action 'Open')
        initial_dist = {s: 0.0 for s in states}
        initial_dist[self._state("S", "Open")] = 0.9
        initial_dist[self._state("E", "Open")] = 0.1

        # Reward Model Formulation: Linear reward over the augmented state space.
        reward_vector = np.zeros(len(states))
        for s in self.disease_states:
            for a in self.actions:
                idx = state_idx[self._state(s, a)]
                reward_vector[idx] = -(self.cost_econ[a] + self.cost_health[s])

        # A single LinearReward serves both the LP (concave) and VI (max-linear) solvers.
        reward_model = [
            LinearReward(
                indices=list(range(len(states))), 
                weights=reward_vector.tolist(), 
                offset=0.0
            )
        ]

        return MDP(
            states=states,
            actions=self.actions,
            state_action_map=state_action_map,
            transition_prob=transition_prob,
            initial_dist=initial_dist,
            reward_model=reward_model,
            is_concave_reward=True, 
            is_max_linear_reward=True 
        )


if __name__ == "__main__":
    generator = SEIR_ICU_ControlMDP()
    mdp = generator.generate_mdp()
    horizon = 10
    gamma = 0.95
    
    print(f"State Space Size: {len(mdp.states)}")
    print(f"Action Space Size: {len(mdp.actions)}")
    print(f"Deterministic Policies: {len(mdp.actions)**len(mdp.states)}")
    print("-" * 50)
    
    # 1. Test Convex Optimization (LP / Concave Pipeline)
    print("Running Convex Optimization (FusionMDPSolver)...")
    start_time = time.time()
    
    lp_solver = FusionMDPSolver(mdp=mdp, horizon=horizon, gamma=gamma)
    lp_policy, lp_obj = lp_solver.solve()
    
    lp_time = time.time() - start_time
    print(f"Optimal Value (LP): {lp_obj:.4f}")
    print(f"Runtime (LP):       {lp_time:.4f} seconds\n")
    
    # 2. Test Value Iteration (VI / Max-Linear Pipeline)
    print("Running Value Iteration (MaxLinearValueIterationSolver)...")
    start_time = time.time()
    
    vi_solver = MaxLinearValueIterationSolver(mdp=mdp, horizon=horizon, gamma=gamma)
    vi_solver.solve()
    vi_obj = vi_solver.value(mdp.initial_dist, t=0)
    
    vi_time = time.time() - start_time
    print(f"Optimal Value (VI): {vi_obj:.4f}")
    print(f"Runtime (VI):       {vi_time:.4f} seconds")