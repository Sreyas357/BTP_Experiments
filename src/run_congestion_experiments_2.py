import argparse
import time
from examples.congestion import *
from examples.congestion_destinations import DESTINATION_SHAPES  
from solvers.convex_opt import FusionMDPSolver  
from src.helper_files.congestion_policy_io import metadata_from_spec, save_policy_json  



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Congestion MDP (LP Solver)")
    parser.add_argument("--size", type=int, default=20, help="Grid size (e.g., 20 for 20x20)")
    parser.add_argument("--shape", type=str, default="smiley", choices=list(DESTINATION_SHAPES.keys()))
    parser.add_argument("--horizon", type=int, default=30, help="Time horizon")
    parser.add_argument("--p_random", type=float, default=0.0, help="Error probability (e.g., 0.1)")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to average")
    parser.add_argument("--timeout",type=float,default=600.0,help="timeout in sec")

    args = parser.parse_args()

    # Setup parameters
    dest_cells = DESTINATION_SHAPES[args.shape](args.size, args.size)
    
    print(f"Running Congestion LP: {args.shape} {args.size}x{args.size}, Horizon={args.horizon}, Noise={args.noise} ({args.runs} runs)...", end=" ", flush=True)

    mdp = generate_congestion_mdp(
            rows=args.rows,
            cols=args.cols,
            dest_cells=dest_cells,
            stochastic=args.p_random > 0.0,
            p_correct= 1 - args.p_random,
            capacity=(1.0/args.rows)
        )
    
    times = []
    for _ in range(args.runs):
        solver = FusionMDPSolver(mdp, horizon=args.horizon, gamma=1 , precision=1e-5 )
        
        start_time = time.time()
        solver.solve()
        times.append(time.time() - start_time)

    avg_time = sum(times) / args.runs
    print(f"Done! Avg Time: {avg_time:.4f} s")

    # Append to results file
    with open("congestion_results.txt", "a") as f:
        # Write header if file is empty
        if f.tell() == 0:
            f.write("Shape        | Size  | Noise | Horizon | Avg Time (s)\n")
            f.write("-" * 60 + "\n")
        f.write(f"{args.shape:<12} | {args.size}x{args.size:<3} | {args.noise:<5} | {args.horizon:<7} | {avg_time:.4f}\n")