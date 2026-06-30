# -*- coding: utf-8 -*-
"""Greedy insertion benchmark under the rolling-horizon setting.

The heuristic plans with the default average speed only. Its returned objective
value is evaluated afterward with the realized request speeds.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


DEPOTS = np.array([
    [-73.9855, 40.7580],
    [-74.0090, 40.7060],
    [-73.9780, 40.7527],
    [-73.9934, 40.7505],
    [-73.9776, 40.7614],
    [-73.9680, 40.7851],
    [-73.9819, 40.7681],
    [-73.9830, 40.7420],
    [-73.9500, 40.7230],
    [-73.9442, 40.6782],
    [-73.8300, 40.7580],
    [-73.8656, 40.7681],
    [-73.7781, 40.6413],
    [-73.8740, 40.7769],
    [-73.9505, 40.8055],
    [-73.9955, 40.7309],
    [-74.0150, 40.7113],
    [-73.9397, 40.7003],
    [-73.9763, 40.6442],
    [-73.9875, 40.7484],
])

K = 20
AVG_V = 12.94
MU_M = 1.25
MU_T = 5.0
STEPS = 20
DEFAULT_INST_PATH = Path("Data/Test/XS-size/XS_a01.parquet")

VEHI_DEPOT = {k: -((k % len(DEPOTS)) + 1) for k in range(K)}


def make_vehicle_depot(fleet_size):
    return {k: -((k % len(DEPOTS)) + 1) for k in range(fleet_size)}

# Rows: early / on-time / late arrival. Columns: ETA action 0 / 1 / 2.
REWARD_COEFS = np.array([
    [0.20, 0.10, 0.05],
    [-0.15, 0.00, -0.15],
    [-0.20, -0.10, -0.05],
])


class Instance:
    def __init__(self, path=DEFAULT_INST_PATH, top_n=None):
        data = pd.read_parquet(path)
        if top_n is not None:
            data = data.head(top_n).copy()
        data = data.reset_index(drop=True)

        t_base = data["pick_early"].min()
        time_cols = ["pick_early", "pick_late", "drop_early", "drop_late"]
        data[time_cols] = (data[time_cols] - t_base).apply(
            lambda col: col.dt.total_seconds() / 3600
        )

        data["id"] = np.arange(len(data), dtype=int)
        data["status"] = True
        self.data = data.copy()


class HeuristicRollingHorizon:
    def __init__(
        self,
        data,
        vehi_depot=None,
        steps=STEPS,
        seed=None,
        min_delta=12.0,
        fleet_size=K,
        step_time_limit=300.0,
    ):
        self.fleet_size = int(fleet_size)
        if vehi_depot is None:
            vehi_depot = make_vehicle_depot(self.fleet_size)

        self.data = data.copy().reset_index(drop=True)
        self.t0 = float(self.data["pick_early"].min())
        self.T = float(self.data["pick_late"].max())
        self.dt = float(self.T - self.t0) / max(steps - 1, 1)
        self.now = self.t0

        self.depot_map = {-(d + 1): DEPOTS[d] for d in range(len(DEPOTS))}
        self.req_map = self.data.set_index("id", drop=False)
        self.reqs = self.req_map.to_dict("index")
        self.rng = np.random.default_rng(seed)

        self.sol = [
            np.array([[vehi_depot[k], -1, self.t0, 0.0]], dtype=float)
            for k in range(self.fleet_size)
        ]
        self.heads = np.zeros(self.fleet_size, dtype=int)
        self.ov = 0.0
        self.ct = 0.0
        self.min_delta = min_delta
        self.step_time_limit = step_time_limit

    def main(self):
        start = time.time()
        while self.now <= self.T:
            deadline = time.time() + self.step_time_limit
            self.time_pass()
            self.update_heads()
            self.greedy_insert(self.get_active_requests(), deadline=deadline)
            self.now += self.dt

        self.ct = time.time() - start
        return self.calculate_ov(mode="posterior"), self.ct

    def greedy_insert(self, reqs, vehi_idx=None, deadline=None):
        if vehi_idx is None:
            vehi_idx = range(self.fleet_size)

        improved = False
        total_delta = 0.0

        for _, req in reqs.iterrows():
            if deadline is not None and time.time() >= deadline:
                break

            req_id = int(req["id"])
            if not bool(self.data.at[req_id, "status"]):
                continue

            best = self.best_insert(req, vehi_idx=vehi_idx, deadline=deadline)
            if best is None or best["delta"] <= self.min_delta:
                continue

            self.apply_insertion(best)
            total_delta += best["delta"]
            improved = True
            self.set_status(req_id, False)

        return improved, total_delta

    def best_insert(self, req, vehi_idx=None, deadline=None):
        options = self.insertion_options(
            req, vehi_idx=vehi_idx, limit=1, deadline=deadline
        )
        return options[0] if options else None

    def insertion_options(self, req, vehi_idx=None, limit=None, deadline=None):
        if vehi_idx is None:
            vehi_idx = range(self.fleet_size)

        req_id = int(req["id"])
        options = []

        for k in vehi_idx:
            if deadline is not None and time.time() >= deadline:
                break

            route = self.sol[k]
            head = int(self.heads[k])
            base_ov = self.route_objective(route, since=head, mode="prior")

            for pos in range(head + 1, len(route) + 1):
                route_base = self.insert_request(route, req_id, 0, pos)
                route_base = self.update_route(route_base, since=head, mode="prior")
                for eta in range(3):
                    route1 = route_base.copy()
                    route1[pos, 1] = eta
                    new_ov = self.route_profit(route1, since=head)
                    delta = new_ov - base_ov
                    options.append({"delta": delta, "k": k, "route": route1})

        options.sort(key=lambda x: x["delta"], reverse=True)
        if limit is not None:
            return options[:limit]
        return options

    def apply_insertion(self, best):
        head = int(self.heads[best["k"]])
        self.sol[best["k"]] = self.update_route(
            best["route"], since=head, mode="posterior"
        )
        self.ov += best["delta"]

    def insert_request(self, route, req_id, eta, pos):
        new_node = np.array([req_id, eta, np.nan, np.nan], dtype=float)
        return np.insert(route.copy(), pos, new_node, axis=0)

    def remove_request(self, req_id):
        loc = self.find_request(req_id)
        if loc is None:
            return False, 0.0

        k, pos = loc
        head = int(self.heads[k])
        route = self.sol[k]
        base_ov = self.route_objective(route, since=head, mode="prior")
        route1 = np.delete(route, pos, axis=0)
        route1_prior = self.update_route(route1, since=head, mode="prior")
        route1_post = self.update_route(route1, since=head, mode="posterior")
        new_ov = self.route_profit(route1_prior, since=head)
        delta = new_ov - base_ov

        self.sol[k] = route1_post
        self.ov += delta
        self.set_status(req_id, True)
        return True, delta

    def find_request(self, req_id):
        for k, route in enumerate(self.sol):
            head = int(self.heads[k])
            hits = np.where(route[head + 1 :, 0].astype(int) == int(req_id))[0]
            if len(hits) > 0:
                return k, int(hits[0] + head + 1)
        return None

    def removal_candidates(self, active_only=True):
        cands = []
        for k, route in enumerate(self.sol):
            head = int(self.heads[k])
            if len(route) <= head + 1:
                continue

            base_ov = self.route_objective(route, since=head, mode="prior")
            for pos in range(head + 1, len(route)):
                req_id = int(route[pos, 0])
                req = self.reqs[req_id]
                if active_only and float(req["pick_late"]) < self.now:
                    continue

                route1 = np.delete(route, pos, axis=0)
                route1 = self.update_route(route1, since=head, mode="prior")
                delta = self.route_profit(route1, since=head) - base_ov
                cands.append((delta, req_id, k, pos))
        return cands

    def time_pass(self):
        for k, route in enumerate(self.sol):
            if route[-1, 2] < self.now:
                self.sol[k][-1, 2] = self.now

    def update_heads(self):
        for k, route in enumerate(self.sol):
            time_arr = route[:, 2]
            head = np.where(np.isfinite(time_arr) & (time_arr <= self.now))[0]
            self.heads[k] = int(head[-1]) if len(head) > 0 else 0

    def get_active_requests(self, sort_fare=True):
        active = self.data[
            (self.data["pick_early"] <= self.now)
            & (self.data["pick_late"] >= self.now)
            & (self.data["status"])
        ].copy()
        if sort_fare and len(active) > 0:
            active = active.sort_values(by="fare", ascending=False)
        return active.reset_index(drop=True)

    def update_route(self, route, since=0, mode="prior", floor_now=True):
        route = route.copy()
        since = int(min(max(since, 0), len(route) - 1))
        if floor_now:
            route[since, 2] = max(float(route[since, 2]), self.now)

        for i in range(since + 1, len(route)):
            req_id = int(route[i, 0])
            req = self.reqs[req_id]

            prev_id = int(route[i - 1, 0])
            prev_coord = self.get_node_coord(prev_id)
            pick_coord = np.array([req["pick_longt"], req["pick_lat"]])

            dis_topick = self.get_distance(prev_coord, pick_coord)
            dis_trip = float(req["trip_dis"])
            speed = AVG_V if mode == "prior" else float(req["speed"])
            if speed <= 0:
                speed = AVG_V

            t_prev = float(route[i - 1, 2])
            t_pick = max(float(req["pick_early"]), t_prev + dis_topick / AVG_V)
            t_drop = t_pick + dis_trip / speed

            route[i, 2] = t_drop
            route[i, 3] = float(route[i - 1, 3]) + dis_topick + dis_trip
        return route

    def route_objective(self, route, since=0, mode="prior", floor_now=True):
        route1 = self.update_route(route, since=since, mode=mode, floor_now=floor_now)
        return self.route_profit(route1, since=since)

    def route_profit(self, route, since=0):
        ov = 0.0
        for i in range(int(since) + 1, len(route)):
            req_id = int(route[i, 0])
            eta = int(route[i, 1])
            req = self.reqs[req_id]

            delta_dis = float(route[i, 3] - route[i - 1, 3])
            delta_time = float(route[i, 2] - route[i - 1, 2])
            coefs = self.eta_coefs(float(route[i, 2]), req)
            ov += float(req["fare"] * (1 + coefs[eta]) - (MU_M * delta_dis + MU_T * delta_time))
        return ov

    def calculate_ov(self, mode="posterior"):
        ov = 0.0
        for route in self.sol:
            ov += self.route_objective(route, since=0, mode=mode, floor_now=False)
        return ov

    def snapshot(self):
        return [route.copy() for route in self.sol], self.data["status"].copy(), float(self.ov)

    def restore(self, state):
        routes, status, ov = state
        self.sol = [route.copy() for route in routes]
        self.data.loc[:, "status"] = status
        self.ov = float(ov)
        self.update_heads()

    def set_status(self, req_id, status):
        self.data.at[int(req_id), "status"] = bool(status)

    def eta_coefs(self, t_drop, req):
        if t_drop < float(req["drop_early"]):
            advance = (float(req["drop_early"]) - t_drop) * 60
            return np.clip(advance * REWARD_COEFS[0], None, 1)
        if t_drop > float(req["drop_late"]):
            lateness = (t_drop - float(req["drop_late"])) * 60
            return np.clip(lateness * REWARD_COEFS[2], -1, None)
        return REWARD_COEFS[1]

    def get_node_coord(self, node_id):
        node_id = int(node_id)
        if node_id < 0:
            return self.depot_map[node_id]
        req = self.reqs[node_id]
        return [req["drop_longt"], req["drop_lat"]]

    @staticmethod
    def get_distance(coord1, coord2):
        longt1, lat1 = np.radians(coord1[0]), np.radians(coord1[1])
        longt2, lat2 = np.radians(coord2[0]), np.radians(coord2[1])
        dlongt = longt1 - longt2
        dlat = lat1 - lat2
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlongt / 2) ** 2
        )
        return float(2 * np.arcsin(np.sqrt(a)) * 6371)


class GreedyInsertion(HeuristicRollingHorizon):
    pass


def main(inst_path=DEFAULT_INST_PATH, steps=STEPS, 
         top_n=None, seed=None, min_delta=5.0, fleet_size=K,
         step_time_limit=300.0):
    ins = Instance(inst_path, top_n=top_n)
    gi = GreedyInsertion(ins.data, steps=steps, 
                         seed=seed, min_delta=min_delta,
                         fleet_size=fleet_size,
                         step_time_limit=step_time_limit)
    ov, ct = gi.main()
    print(f"Objective value: {ov:.3f} | Runtime: {ct:.3f} sec.")
    return ov, ct


INST_DIR = Path('Data/Test/XS-size')
INSTS = ['XS_a01.parquet']

if __name__ == "__main__":
    for ins in INSTS:
        main(inst_path=INST_DIR / ins)
