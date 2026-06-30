Author: Hao Li, on Jun 30

# RL-agent with ETA-awareness
This project presents example data, core codes, trained assets, and benchmark sources for the study of a ride-hailing dispatching RL agent with ETA-awareness.

## Repository Contents
- `asset/`: saves trained agents and model parameters.
    - `trained agent.pth`: the trained A2C-ARS agent used in this study.
- `benchmarks/`: benchmark source codes used for comparison.
    - `Exact (Gurobi)/Exact.py`: exact optimization benchmark supported by Gurobi.
    - `Heuristics/`: heuristic benchmarks, including GI, ALNS, VNS, and BTA.
    - `Other RL/`: other RL benchmarks, including PPO and VPG.
- `data/Training/`: instances used in training. 
- `data/Test/`: instances used in test. 

> XS-size (≤10), VS-size (50), VS-size (100), M-size (100), and L-size (500) are used to distinguish instances based on request scale. 

- `src/`: core A2C-ARS codes.
    - `training.py`: training file to train agent on instances.
    - `test.py`: test file to run the test instances

## Environment
Please check the necessary libraries and compatible versions:
- `Python 3.10.20`
- `numpy 2.2.6`
- `pandas 2.2.2`
- `torch 2.5.1`
- `scipy 1.15.2`
- `pyarrow 15.0.2` or `fastparquet 2025.12.0`: needed for reading .parquet files.
- `gurobipy 12.0.3`: (optional), only needed for exact benchmark under `Exact (Gurobi)/Exact.py`. 
- `matplotlib 3.9.1`: (optional), only needed for visualization.

Please use the project root as the working directory on the computer:

```bash
RL-agent with ETA-awareness
```

, codes in this project use relative paths based on this working directory. 

Recommended tools:
- `VSCode`: IDE for working on the project.
- `Codex`: auxiliary AI tool to explain, review, and revise the codes.

## Usage
### train a new agent 
Run the training file:

```bash
python src/training.py
```

, and the trained parameters should be found in `asset`. 

We suggest using instances under `data/Training/VS-size` as training set. Their scale is suitable, not requiring for substantial runtime, which can help fast validate some results. 

### test a trained agent 
Run the test file: 

```bash
python src/test.py
```

Serveral instances in various sizes, aligned with the selected instances in the paper, are placed under `data/Test/VS-size`. 

## Notes
- Some source files contain non-ASCII variable names (e.g., Δ). Please ensure the files are encoded in `UTF-8`. 
- Since upload size limit, we can only online upload some example data in this repository. Please contact us at `hawklee0527@outlook.com` if needed to access the full dataset. 
- Other problems are also welcomed. 

## License
NOTE: if you would like to run `Exact (Gurobi)/Exact.py` for exact solutions, you should first request for an official licence from Gurobi.

