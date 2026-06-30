# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 20:49:43 2025

@author: MR
"""

import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum, LinExpr

initial_depots = np.array([
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
    [-74.0150, 40.7113],  # World Trade CenterG
    [-73.9397, 40.7003],  # Bushwick (Brooklyn)
    [-73.9763, 40.6442],  # Prospect Park South
    [-73.9875, 40.7484]   # Empire State Building
])

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# ====================
# Calculation Function
# ====================

def get_sphere_distance(longt1, lat1, longt2, lat2):
    longt1 = np.radians(longt1)
    lat1 = np.radians(lat1)
    longt2 = np.radians(longt2)
    lat2 = np.radians(lat2)

    Δlongt = longt1 - longt2
    Δlat = lat1 - lat2

    a = (np.sin(Δlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(Δlongt / 2) ** 2)
    return 2 * np.arcsin(np.sqrt(a)) * 6371

def get_topick_distance(req, O):
    O_arr = np.array(list(O.values()))
    dataframe = pd.DataFrame({
        'vehi_longt': O_arr[:, 0], 'vehi_lat': O_arr[:, 1],
        'pick_longt': req['pick_longt'], 'pick_lat': req['pick_lat']})
    
    dataframe['trip_dis'] = get_sphere_distance(
        dataframe['vehi_longt'].values, dataframe['vehi_lat'].values,
        dataframe['pick_longt'].values, dataframe['pick_lat'].values)
    return dataframe['trip_dis'].values

# ====================
# Gurobi solving
# ====================

if __name__ == '__main__':
    path = 'Data/Test/XS-size/XS_a01.parquet'
    instance_slice = slice(0, 20)
    static_version = True
    enforce_pick_late = False
    
    # load instance
    inst = pd.read_parquet(path)[instance_slice]
    inst = inst.reset_index().rename(columns={'index': 'req_id'})
    inst['status'] = True
    
    # parameters
    req_size = len(inst)
    fleet_size = 4      # fleet size
    ETA_actions = 3     # 0 == advance; 1 == normal; 2 == normal
    avg_speed = 12.94   #inst['speed'].mean()
    M = 1e4
    
    μm = 1.25; μt = 5.0
    # Rows: early / on-time / late arrival. Columns: ETA action 0 / 1 / 2.
    # Keep this aligned with src/training.py.
    reward_coefs = np.array([
        [0.20, 0.10, 0.05],
        [-0.15, 0.00, -0.15],
        [-0.20, -0.10, -0.05],
    ])
    advance_coef = reward_coefs[0]
    normal_coef = reward_coefs[1]
    lateness_coef = reward_coefs[2]
    
    # sets
    O = {k: initial_depots[k % len(initial_depots)] + np.random.uniform(-0, 0, 2)
         for k in range(fleet_size)}
    Ot = {k: inst['pick_early'].min() + pd.Timedelta(np.random.uniform(-0, 0), unit='min')             
         for k in range(fleet_size)}
    Cko = np.array([get_topick_distance(req, O) for i, req in inst.iterrows()]).T  # [vehicle, req_index]
    
    Cdo = get_sphere_distance(inst['drop_longt'].values[:, None], inst['drop_lat'].values[:, None], 
                              inst['pick_longt'].values[None, :], inst['pick_lat'].values[None, :])   # [req1_index, req2_index]
    Cod = inst['trip_dis'].values
    Tod = inst['trip_dis'].values / inst['speed'].values

    P = range(req_size)
    D = range(req_size, 2*req_size)
    depot = range(2*req_size, 2*req_size + fleet_size)
    dummy = range(2*req_size + fleet_size, 2*(req_size + fleet_size))
    V = range(2*(req_size + fleet_size))
    
    K = range(fleet_size)
    I = range(req_size)
    E = range(ETA_actions)
 
    F = inst['fare'].values
    t_base = inst['pick_early'].min()
    pick_early = ((inst['pick_early'] - t_base).dt.total_seconds() / 3600).values
    pick_late = ((inst['pick_late'] - t_base).dt.total_seconds() / 3600).values
    drop_early = ((inst['drop_early'] - t_base).dt.total_seconds() / 3600).values
    drop_late = ((inst['drop_late'] - t_base).dt.total_seconds() / 3600).values
    if static_version:
        pick_early = np.zeros_like(pick_early)
    
    # model
    mdl = Model()
    # decision variables
    x = mdl.addVars(K, V, V, vtype=GRB.BINARY)
    y = mdl.addVars(K, I, vtype=GRB.BINARY)
    e = mdl.addVars(I, E, vtype=GRB.BINARY)
    eta_normal = mdl.addVars(I, E, vtype=GRB.BINARY)
    t = mdl.addVars(V, vtype=GRB.CONTINUOUS, lb=0)
    u = mdl.addVars(V, vtype=GRB.CONTINUOUS, lb=0, ub=len(V)+1)
    
    # (1) request is serviced by one vehicle at most
    for i in I:
        mdl.addConstr(quicksum(y[k,i] for k in K) <= 1)
    
    # (2) flow balance with depots
    for k in K:
        # (2.1.1) only an exit from its depot
        mdl.addConstr(quicksum(x[k,depot[k],j] for j in V if j != depot[k]) == 1)
        for dp in depot:
            if dp != depot[k]:
                mdl.addConstr(quicksum(x[k,dp,j] for j in V if j != dp) == 0)
    
        # (2.2) only an entry into its dummy
        mdl.addConstr(quicksum(x[k,j,dummy[k]] for j in V if j != dummy[k]) == 1)
        for dy in dummy:
            if dy != dummy[k]:
                mdl.addConstr(quicksum(x[k,j,dy] for j in V if j != dy) == 0)
        # (2.3) no outflows for dummy
        for dy in dummy:
            mdl.addConstr(quicksum(x[k,dy,j] for j in V if j != dy) == 0)
            
    
    # (2.2) basic flow
    for k in K:
        for i in (list(P) + list(D)):
            mdl.addConstr(quicksum(x[k,i,j] for j in V if i != j) == 
                          quicksum(x[k,j,i] for j in V if i != j))
    
    # (2.3) drop must be consective to pick, no ride-sharing
    for k in K:
        for i in I:
            mdl.addConstr(quicksum(x[k,j,P[i]] for j in V if j != P[i]) == y[k,i])
            mdl.addConstr(x[k,P[i],D[i]] == y[k,i])
    
    # (2.4) pick and drop have identical visits
    for k in K:
        for i in I:
            mdl.addConstr(quicksum(x[k,j,P[i]] for j in V if j != P[i]) == 
                          quicksum(x[k,j,D[i]] for j in V if j != D[i]))  
            
    # (2.5) MTZ subloop elimination
    for k in K:
        for i in (list(P) + list(D)):  
            for j in (list(P) + list(D)):
                if i != j:
                    mdl.addConstr(u[j] >= u[i] + 1 - M*(1 - x[k,i,j]))

    # (3) time constraints
    # (3.1.1) depot to outside
    for k in K:
        for i in I:
            tko = Cko[k,i] / avg_speed
            mdl.addConstr(t[P[i]] >= t[depot[k]] + tko - M*(1 - x[k,depot[k],P[i]]))
    # (3.1.2) pick to drop
            tod = Tod[i]
            mdl.addConstr(t[D[i]] >= t[P[i]] + tod - M*(1 - x[k,P[i],D[i]]))
    # (3.1.3) to dummy
            mdl.addConstr(t[dummy[k]] >= t[D[i]] - M*(1 - x[k,D[i],dummy[k]]))
    # (3.1.4) drop to next pick
    for k in K:
        mdl.addConstr(t[depot[k]] == 0)
        for i in I:
            for j in I:
                if i != j:
                    tdo = Cdo[i,j] / avg_speed
                    mdl.addConstr(t[P[j]] >= t[D[i]] + tdo - M*(1 - x[k,D[i],P[j]]))
    # (3.2)
    for i in I:
        mdl.addConstr(t[D[i]] >= t[P[i]] + 1/M)

    # (3.3) time window
    for i in I:
        served_i = quicksum(y[k,i] for k in K)
        mdl.addConstr(t[P[i]] >= pick_early[i] - M*(1 - served_i))
        if enforce_pick_late:
            mdl.addConstr(t[P[i]] <= pick_late[i] + M*(1 - served_i))
        
    # (4) profit definition
    revenue_expr = LinExpr()
    cost_expr = LinExpr()

    
    
    # (4.0.1) auxiliary variable
    advance = mdl.addVars(I, vtype=GRB.BINARY)  
    normal = mdl.addVars(I, vtype=GRB.BINARY)   
    lateness = mdl.addVars(I, vtype=GRB.BINARY)
    for i in I:
        mdl.addConstr(advance[i] + normal[i] + lateness[i] == \
                      quicksum(y[k,i] for k in K))
        mdl.addConstr(t[D[i]] <= drop_early[i] + M*(1 - advance[i]))
        mdl.addConstr(t[D[i]] >= drop_late[i] - M*(1 - lateness[i]))
        mdl.addConstr(t[D[i]] >= drop_early[i] - M*(1 - normal[i]))
        mdl.addConstr(t[D[i]] <= drop_late[i] + M*(1 - normal[i]))
    
    
    advance_minute = mdl.addVars(I, vtype=GRB.CONTINUOUS, lb=0)
    lateness_minute = mdl.addVars(I, vtype=GRB.CONTINUOUS, lb=0)

    for i in I:
        mdl.addConstr(advance_minute[i] <= 60*(drop_early[i] - t[D[i]]) + M*(1-advance[i]))
        mdl.addConstr(advance_minute[i] <= M*advance[i])
        
        mdl.addConstr(lateness_minute[i] >= 60*(t[D[i]] - drop_late[i]) - M*(1-lateness[i]))
        mdl.addConstr(lateness_minute[i] <= M*lateness[i])

    z_advance = mdl.addVars(I, E, vtype=GRB.CONTINUOUS, lb=0)
    z_lateness = mdl.addVars(I, E, vtype=GRB.CONTINUOUS, lb=0)
    coef_raw = mdl.addVars(I, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    for i in I:
        for eta in E:
            mdl.addConstr(z_advance[i,eta] <= advance_minute[i])
            mdl.addConstr(z_advance[i,eta] <= M*e[i,eta])
            mdl.addConstr(z_advance[i,eta] >= advance_minute[i] - M*(1 - e[i,eta]))
    
            mdl.addConstr(z_lateness[i,eta] <= lateness_minute[i])
            mdl.addConstr(z_lateness[i,eta] <= M*e[i,eta])
            mdl.addConstr(z_lateness[i,eta] >= lateness_minute[i] - M*(1 - e[i,eta]))

            mdl.addConstr(eta_normal[i,eta] <= normal[i])
            mdl.addConstr(eta_normal[i,eta] <= e[i,eta])
            mdl.addConstr(eta_normal[i,eta] >= normal[i] + e[i,eta] - 1)
    
        mdl.addConstr(
            coef_raw[i] ==
            quicksum(advance_coef[eta] * z_advance[i,eta] for eta in E) +
            quicksum(lateness_coef[eta] * z_lateness[i,eta] for eta in E) +
            quicksum(normal_coef[eta] * eta_normal[i,eta] for eta in E))
    
    z1 = mdl.addVars(I, vtype=GRB.BINARY)   # raw >= 1?
    z2 = mdl.addVars(I, vtype=GRB.BINARY)   # raw <= -1?
    coef = mdl.addVars(I, lb=-1, ub=1, vtype=GRB.CONTINUOUS)
    for i in I:
        mdl.addConstr(z1[i] + z2[i] <= 1)
        
        # z1=1 means coef_raw >= 1 -> coef = 1
        mdl.addConstr(coef_raw[i] >= 1 - M*(1-z1[i]))
        mdl.addConstr(coef[i] >= 1 - M * (1 - z1[i]))
        mdl.addConstr(coef[i] <= 1 + M * (1 - z1[i]))
        
        # z2=1 means coef_raw <= -1 -> coef = -1
        mdl.addConstr(coef_raw[i] <= -1 + M*(1-z2[i]))
        mdl.addConstr(coef[i] >= -1 - M * (1 - z2[i]))
        mdl.addConstr(coef[i] <= -1 + M * (1 - z2[i]))
        
        # else clip interior: z1=0 & z2=0 -> coef = coef_raw
        mdl.addConstr(coef[i] >= coef_raw[i] - M * (z1[i] + z2[i]))
        mdl.addConstr(coef[i] <= coef_raw[i] + M * (z1[i] + z2[i]))


    # (4.0.2) only given one ETA level if the request is served
    for i in I:
        mdl.addConstr(quicksum(e[i, eta] for eta in E) == quicksum(y[k,i] for k in K))

    # (4.1) revenue
    for k in K:
        for i in I:
            revenue_expr += y[k,i] * F[i]
    for i in I:
        revenue_expr += F[i] * coef[i]

    # (4.2) cost
    for k in K:
        for i in I:
        # (4.2.1) depot to out
            mile_ko = Cko[k,i]
            hour_ko = Cko[k,i] / avg_speed
            cost_ko = μm * mile_ko + μt * hour_ko
            cost_expr -= x[k,depot[k],P[i]] * cost_ko
        # (4.2.2) pick to drop
            mile_od = Cod[i]
            hour_od = Tod[i]
            cost_od =  μm*mile_od + μt*hour_od
            cost_expr -= x[k,P[i],D[i]] * cost_od
        # (4.2.3) drop to next pick
            for j in I:
                if i != j:
                    mile_do = Cdo[i,j]
                    hour_do = Cdo[i,j] / avg_speed
                    cost_do = μm*mile_do + μt*hour_do
                    cost_expr -= x[k,D[i],P[j]] * cost_do

    # objective function
    mdl.setObjective(revenue_expr + cost_expr, GRB.MAXIMIZE)
    mdl.setParam('TimeLimit', GRB.INFINITY)      
    #mdl.setParam('MIPGap', 0.01)                           
    #mdl.setParam('MIPFocus', 0)
    
    mdl.optimize()
