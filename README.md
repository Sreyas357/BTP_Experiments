# Create venv
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\actate

# Upgrade pip
pip install --upgrade pip


# To run Congestion expermients ( Concave reward one )
usage: run_congestion_experiments.py [-h] [--sizes SIZES [SIZES ...]]
                                     [--shapes {h_center,iitb,hello_world,smiley} [{h_center,iitb,hello_world,smiley} ...]]
                                     --horizons HORIZONS [HORIZONS ...] [--gamma GAMMA] [--capacity CAPACITY]
                                     [--move-cost MOVE_COST] [--congestion-penalty CONGESTION_PENALTY]
                                     [--noise NOISE [NOISE ...]] [--randomized] [--timeout TIMEOUT] [--jobs JOBS]
                                     [--mosek-threads MOSEK_THREADS] [--precision PRECISION]
                                     [--policy-dir POLICY_DIR]

# To run path concentration examples ( the value iteration ones )

python3 src/run_tests_vi.py --help
usage: run_tests_vi.py [-h] [--rows ROWS] [--cols COLS] [--horizon HORIZON] [--gamma GAMMA] [--time-penalty TIME_PENALTY]
                       [--concentration-penalty CONCENTRATION_PENALTY] [--penalty-states-cost PENALTY_STATES_COST]
                       [--num-random-penalty-states NUM_RANDOM_PENALTY_STATES] [--stochastic] [--p-correct P_CORRECT] [--timeout TIMEOUT]
                       [--pbvi-max-vectors PBVI_MAX_VECTORS] [--pbvi-belief-points PBVI_BELIEF_POINTS] [--seed SEED] [--out-dir OUT_DIR] [--worker]
                       [--method {vi_exact,vi_pbvi}] [--output OUTPUT]

# To run the epidemic example 

python3 src/run_sirv_epidemic_tests.py
