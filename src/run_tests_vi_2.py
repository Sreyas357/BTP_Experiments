import argparse
import time
import multiprocessing  

from examples.path_concentration import *
from solvers.value_iteration import MaxLinearValueIterationSolver
from src.helper_files.congestion_policy_io import save_policy_json


def solve_task(mdp,args):
    if args.use_pbvi: 
        solver = MaxLinearValueIterationSolver(mdp, horizon=args.horizon, gamma=1 , max_vectors=500)
    else:
        solver = MaxLinearValueIterationSolver(mdp, horizon=args.horizon, gamma=1)

    solver.solve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Warehouse Path Concentration (VI Solver)")
    parser.add_argument("--rows", type=int, default=2, help="Number of rows")
    parser.add_argument("--cols", type=int, default=5, help="Number of columns")
    parser.add_argument("--horizon", type=int, default=12, help="Time horizon")
    parser.add_argument("--p_random", type=float, default=0.0, help="Error probability (e.g., 0.1)")
    parser.add_argument("--use_pbvi", action="store_true")
    parser.add_argument("--timeout",type=float,default=600.0,help="timeout in sec")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to average")
    parser.add_argument("--gamma", type=int, default=1, help="gamma")

    
    args = parser.parse_args()

    print(f"Running Warehouse VI: {args.rows}x{args.cols}, Horizon={args.horizon} ({args.runs} runs)...", end=" ", flush=True)

    mdp,_,_,_ = generate_warehouse_path_concentration_mdp(
        rows=args.rows,
        cols=args.cols,
        is_stochastic=args.p_random > 0.0,
        p_correct= 1 - args.p_random
    )

    times = []
    timeouts = 0
    avg_time = 0
    
    for _ in range(args.runs):
        
        p = multiprocessing.Process(target=solve_task, args=(mdp, args))
        
        start_time = time.time()
        p.start()
        p.join(args.timeout)
        
        if p.is_alive():
            p.terminate()
            p.join()
            timeouts += 1
        else:
            times.append(time.time() - start_time)
            
    if timeouts == args.runs:
        print(f"FAILED (Timed out after {args.timeout}s)")
    else:
        avg_time = sum(times) / len(times)
        avg_time_str = f"{avg_time:.4f}"
        print(f"Done! Avg Time: {avg_time_str} s (Timeouts: {timeouts})")

   
    # Append to results file
    with open("vi_results.txt", "a") as f:
        # Write header if file is empty
        if f.tell() == 0:
            f.write("Rows x Cols | Horizon | Avg Time (s)\n")
            f.write("-" * 40 + "\n")
        f.write(f"{args.rows}x{args.cols:<8} | {args.horizon:<7} | {avg_time:.4f}\n")