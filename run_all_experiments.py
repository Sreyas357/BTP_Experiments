import subprocess
import re
import sys

def run_experiments():
    # Define the experiment parameters
    shapes = [ "smiley"]
    grid_sizes = [40]
    
    # Dynamics represented as: (is_stochastic_flag, noise_value)
    dynamics_configs = [
        (False, 0.0), # Deterministic
        (True, 0.1)   # Stochastic
    ]
    
    num_runs = 3
    output_file = "experiment_results_log.txt"

    # Initialize the output file with a clean header
    with open(output_file, "w") as f:
        f.write(f"{'Shape':<12} | {'Size':<7} | {'Noise':<5} | {'Horizon':<7} | {'Avg Time (s)'}\n")
        f.write("-" * 60 + "\n")

    for shape in shapes:
        for size in grid_sizes:
            for is_stochastic, noise in dynamics_configs:
                
                # Calculate horizon based on row count
                horizon = size + 10
                size_str = f"{size}x{size}"
                times = []
                
                print(f"\nRunning -> Shape: {shape}, Size: {size_str}, Noise: {noise}, Horizon: {horizon}")
                
                for run_idx in range(num_runs):
                    print(f"  Run {run_idx + 1}/{num_runs}... ", end="", flush=True)
                    
                    # Construct the CLI command for your existing script
                    cmd = [
                        sys.executable, "src/run_congestion_experiments.py",
                        "--horizons", str(horizon),
                        "--sizes", size_str,
                        "--shapes", shape
                    ]
                    
                    if is_stochastic:
                        cmd.extend(["--randomized", "--noise", str(noise)])
                        
                    try:
                        # Execute the script
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        
                        # Parse the runtime from the summary block: "time_s (min/med/max): 44.008 / 44.008 / 44.008"
                        match = re.search(r"time_s\s*\(min/med/max\):\s*([\d\.]+)", result.stdout)
                        
                        if match:
                            run_time = float(match.group(1))
                            times.append(run_time)
                            print(f"{run_time:.3f} s")
                        else:
                            print(f"FAILED to parse time. Script output snippet:\n{result.stdout[-250:]}")
                            
                    except subprocess.CalledProcessError as e:
                        print(f"CRASHED. Exit Code: {e.returncode}")
                
                # Compute the average and append to the log file
                if times:
                    avg_time = sum(times) / len(times)
                    print(f"  => Average Time: {avg_time:.3f} s")
                    
                    with open(output_file, "a") as f:
                        f.write(f"{shape:<12} | {size_str:<7} | {noise:<5} | {horizon:<7} | {avg_time:.3f}\n")
                else:
                    print("  => All runs failed for this configuration.")
                    with open(output_file, "a") as f:
                        f.write(f"{shape:<12} | {size_str:<7} | {noise:<5} | {horizon:<7} | FAILED\n")

if __name__ == "__main__":
    print("Beginning Congestion Experiment Suite...")
    run_experiments()
    print("\nAll experiments finished. Results saved to 'experiment_results_log.txt'.")