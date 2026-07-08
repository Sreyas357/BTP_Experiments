import time
import numpy as np
from typing import Dict, List, Tuple

from mdp.mdp import MDP
from mdp.reward import LinearReward
from solvers.convex_opt import FusionMDPSolver
from solvers.value_iteration import MaxLinearValueIterationSolver

class SIRV_ControlMDP:
    """
    Distributional MDP for Epidemic Control using the SIRV model.
    Includes Susceptible, Infectious, Recovered, and two Vaccine dose states.
    """
    def __init__(
        self,
        # Transmission rates (linearized for Distributional MDP)
        beta_open: float = 0.20,
        beta_intervene: float = 0.05, 
        rho: float = 0.50,            # V1 relative transmission risk 
        v1_rate: float = 0.10,        # 1st dose administration rate
        v2_rate: float = 0.10,        # 2nd dose administration rate
        gamma: float = 0.10,          # I -> R recovery rate
        nu: float = 0.01,             # R -> S immunity loss rate 
        
        # Costs from Table 3 
        cost_mask_per_day: float = 0.05,
        cost_vaccine_dose: float = 40.0,
        cost_infectious: float = 173.0,
    ):
        self.disease_states = ["S", "I", "R", "V1", "V2"]
        # Actions simplified to 'Open' vs a combined NPI 'Intervene' (Masks + Vaccines)
        self.actions = ["Open", "Intervene"]
        
        self.beta = {"Open": beta_open, "Intervene": beta_intervene}
        self.rho = rho
        self.v1 = v1_rate
        self.v2 = v2_rate
        self.gamma = gamma
        self.nu = nu
        
        self.cost_mask = cost_mask_per_day
        self.cost_vaccine = cost_vaccine_dose
        self.cost_infectious = cost_infectious

    def _state(self, s: str, a: str) -> str:
        # Augment state space S' = S x A to ensure reward is solely a function of state distribution
        return f"{s}_{a}"

    def generate_mdp(self) -> MDP:
        states = [self._state(s, a) for s in self.disease_states for a in self.actions]
        state_idx = {s: i for i, s in enumerate(states)}
        state_action_map = {state: self.actions for state in states}
        
        transition_prob = {}
        for s_prev in self.disease_states:
            for a_prev in self.actions:
                curr = self._state(s_prev, a_prev)
                
                for a_new in self.actions:
                    b_rate = self.beta[a_new]
                    
                    if s_prev == "S":
                        v_rate = self.v1 if a_new == "Intervene" else 0.0
                        transition_prob[(curr, a_new, self._state("I", a_new))] = b_rate
                        transition_prob[(curr, a_new, self._state("V1", a_new))] = v_rate
                        transition_prob[(curr, a_new, self._state("S", a_new))] = 1.0 - b_rate - v_rate
                        
                    elif s_prev == "V1":
                        v_rate = self.v2 if a_new == "Intervene" else 0.0
                        i_rate = b_rate * self.rho
                        transition_prob[(curr, a_new, self._state("I", a_new))] = i_rate
                        transition_prob[(curr, a_new, self._state("V2", a_new))] = v_rate
                        transition_prob[(curr, a_new, self._state("V1", a_new))] = 1.0 - i_rate - v_rate
                        
                    elif s_prev == "I":
                        transition_prob[(curr, a_new, self._state("R", a_new))] = self.gamma
                        transition_prob[(curr, a_new, self._state("I", a_new))] = 1.0 - self.gamma
                        
                    elif s_prev == "R":
                        transition_prob[(curr, a_new, self._state("S", a_new))] = self.nu
                        transition_prob[(curr, a_new, self._state("R", a_new))] = 1.0 - self.nu
                        
                    elif s_prev == "V2":
                        transition_prob[(curr, a_new, self._state("V2", a_new))] = 1.0
        
        initial_dist = {s: 0.0 for s in states}
        initial_dist[self._state("S", "Open")] = 0.99
        initial_dist[self._state("I", "Open")] = 0.01

        # Calculate distributional reward vector based on per-capita costs
        reward_vector = np.zeros(len(states))
        for s in self.disease_states:
            for a in self.actions:
                idx = state_idx[self._state(s, a)]
                cost = 0.0
                
                if s == "I":
                    cost += self.cost_infectious
                
                if a == "Intervene":
                    cost += self.cost_mask
                    # Expected cost of administering vaccines to the mass currently in eligible compartments
                    if s == "S":
                        cost += self.v1 * self.cost_vaccine
                    elif s == "V1":
                        cost += self.v2 * self.cost_vaccine
                        
                reward_vector[idx] = -cost

        reward_model = [
            LinearReward(indices=list(range(len(states))), weights=reward_vector.tolist(), offset=0.0)
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
