import time
import numpy as np
from typing import Dict, List, Tuple

from mdp.mdp import MDP
from mdp.reward import LinearReward
from solvers.convex_opt import FusionMDPSolver
from solvers.value_iteration import MaxLinearValueIterationSolver
from examples.sirv_epidemic_test import *

if __name__ == "__main__":
    generator = SIRV_ControlMDP()
    mdp = generator.generate_mdp()
    gamma = 0.95
    horizons = [10, 20, 40, 80, 160,320,640]
    
    print(f"State Space Size: {len(mdp.states)}")
    print(f"Action Space Size: {len(mdp.actions)}")
    print(f"Deterministic Policies per step: {len(mdp.actions)**len(mdp.states)}")
    print("-" * 80)
    
    print(f"{'Horizon':<8} | {'LP Value':<12} | {'VI Value':<12} | {'LP Time (s)':<15} | {'VI Time (s)':<15}")
    print("-" * 80)

    for h in horizons:
        # --- Convex Optimization (LP) ---
        start_time = time.time()
        lp_solver = FusionMDPSolver(mdp=mdp, horizon=h, gamma=gamma)
        _, lp_obj = lp_solver.solve()
        lp_time = time.time() - start_time
        
        # --- Value Iteration (VI) ---
        start_time = time.time()
        vi_solver = MaxLinearValueIterationSolver(mdp=mdp, horizon=h, gamma=gamma)
        vi_solver.solve()
        vi_obj = vi_solver.value(mdp.initial_dist, t=0)
        vi_time = time.time() - start_time
        
        print(f"{h:<8} | {lp_obj:<12.4f} | {vi_obj:<12.4f} | {lp_time:<15.4f} | {vi_time:<15.4f}")