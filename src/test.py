# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:02:25 2026

@author: MR
"""

import numpy as np
import os
import training as trn
import time

# ======================
# parameters (default)
# ======================
depots = np.array([
    [-73.9855, 40.7580],  # Times Square
    [-74.0090, 40.7060],  # Wall Street / Financial District
    [-73.9780, 40.7527],  # Grand Central Terminal
    [-73.9934, 40.7505],  # Penn Station
    [-73.9776, 40.7614],  # Rockefeller Center
    [-73.9680, 40.7851],  # Central Park East
    [-73.9819, 40.7681],  # Columbus Circle
    [-73.9830, 40.7420],  # Chelsea
    [-73.9500, 40.7230],  # Williamsburg (Brooklyn)
    [-73.9442, 40.6782],  # Downtown Brooklyn
    [-73.8300, 40.7580],  # Flushing (Queens)
    [-73.8656, 40.7681],  # Jackson Heights
    [-73.7781, 40.6413],  # JFK Airport
    [-73.8740, 40.7769],  # LaGuardia Airport
    [-73.9505, 40.8055],  # Harlem
    [-73.9955, 40.7309],  # Greenwich Village
    [-74.0150, 40.7113],  # World Trade Center
    [-73.9397, 40.7003],  # Bushwick (Brooklyn)
    [-73.9763, 40.6442],  # Prospect Park South
    [-73.9875, 40.7484]   # Empire State Building
])
avg_v = 12.94                   # default average speed (according to statistic)
longt0 = -73.96; lat0 = 40.74   # normalized coordinate center
μm = 1.25; μt = 5.0             # distance-to-cost, time-to-cost coefficients

# ETA-impact factor matrix
# ETA → early, on-time, late    
reward_coefs = np.array([ # ATA ↓
    [0.2, 0.10, 0.05],    # early arrival
    [-0.15, 0.0, -0.15],    # on-time arrival
    [-0.2, -0.10, -0.05]  # late arrival
])

# ======================
# important input parameters
# ======================
K = 2                          # fleet size
steps = 1                      # optimization steps
vehi_depot = {k:-((k%len(depots)) + 1) for k in range(K)}  # vehicle depot

(trn.depots, trn.K, trn.avg_v, trn.longt0, trn.lat0, 
 trn.μm, trn.μt, trn.steps, trn.vehi_depot, trn.reward_coefs) = \
    (depots, K, avg_v, longt0, lat0, μm, μt, steps, vehi_depot, reward_coefs)

def main(agent_path='asset/trained agent.pth', 
         ins_path='Data/Test/XS-size/XS_a01.parquet', 
         ins_slice=slice(None), modes=['sample', 'greedy'], trys=1): 
    RL = trn.Learning(max_episode=2000, γ=0.5, reward_scale=0.01, 
                      τ=0.0, τ_max=10.0, τ_min=0.0, reject_loss=-0, 
                      lr_actor=1e-4, lr_critic=1e-4)
    
    # load a trained agent 
    RL.load_model(load_path=agent_path)

    # load the tested instance
    ins = trn.Instance(path=ins_path, ins_slice=ins_slice, static_version=True)
    
    # there are two test modes: 
    # sample: sampling candidates based on logits; 
    # greedy: always sampling the one with the largest logit (recommended)
    print(f'Instance {ins_path} Size {len(ins.data)} Steps {steps}')
    if 'sample' in modes:
        start = time.time()
        returns = RL.evaluate(data=ins.data, trys=trys, greedy_sample=False)
        end = time.time()
        print('test mode: sample')
        print(f'obj. value: {returns.mean():.3f}')
        print(f'runtime: {(end - start):.3f}')

    if 'greedy' in modes:
        start = time.time()
        returns = RL.evaluate(data=ins.data, trys=trys, greedy_sample=True)
        end = time.time()
        print('test mode: greedy')
        print(f'obj. value: {returns.mean():.3f}')
        print(f'runtime: {(end - start):.3f}')

if __name__ == '__main__': 
    main(agent_path='asset/trained agent.pth',                # specify trained agent
         ins_path='Data/Test/XS-size/XS_a01.parquet',         # specify tested instance
         ins_slice=slice(None),                               # slice instance (option)
         trys=1,                                              # how many runs
         modes=['sample', 'test']                             # used test modes
         )
