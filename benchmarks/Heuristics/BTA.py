# -*- coding: utf-8 -*-
"""Batch trip-assignment benchmark for the rolling-horizon ridesharing problem.

The implementation borrows the assignment-optimality idea from recent online
ridesharing benchmarks: generate several feasible trip candidates per vehicle,
then solve a small set-packing assignment problem for the current batch.

By default the online search scores candidates with the same average-speed
prior used by the other heuristic baselines. The final reported objective is
still evaluated with realized speeds for benchmark consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path

import numpy as np

try:
    from scipy import sparse
    from scipy.optimize import Bounds, LinearConstraint, milp
except Exception:  # pragma: no cover - fallback is kept for lean environments.
    sparse = None
    Bounds = LinearConstraint = milp = None

try:
    from .GI import (
        DEFAULT_INST_PATH,
        DEPOTS,
        K,
        STEPS,
        Instance,
        HeuristicRollingHorizon,
    )
except ImportError:
    from GI import (
        DEFAULT_INST_PATH,
        DEPOTS,
        K,
        STEPS,
        Instance,
        HeuristicRollingHorizon,
    )


def make_vehicle_depot(fleet_size):
    return {k: -((k % len(DEPOTS)) + 1) for k in range(fleet_size)}



@dataclass
class TripCandidate:
    k: int
    req_ids: tuple[int, ...]
    route: np.ndarray
    delta: float


class BatchTripAssignment(HeuristicRollingHorizon):
    def __init__(
        self,
        data,
        steps=STEPS,
        seed=None,
        max_bundle_size=4,
        beam_width=14,
        vehicle_pool_cap=22,
        candidate_cap=45,
        repair_rounds=1,
        local_iter=0,
        local_candidate_cap=8,
        insert_min_delta=None,
        candidate_min_delta=None,
        min_delta=10.0,
        score_mode="prior",
        reassign_active=False,
        milp_time_limit=2.0,
        fleet_size=K,
        step_time_limit=300.0,
    ):
        self.fleet_size = int(fleet_size)
        super().__init__(
            data,
            steps=steps,
            seed=seed,
            min_delta=min_delta,
            fleet_size=self.fleet_size,
            step_time_limit=step_time_limit,
        )

        self.max_bundle_size = max_bundle_size
        self.beam_width = beam_width
        self.vehicle_pool_cap = vehicle_pool_cap
        self.candidate_cap = candidate_cap
        self.repair_rounds = repair_rounds
        self.local_iter = local_iter
        self.local_candidate_cap = local_candidate_cap
        self.insert_min_delta = min_delta if insert_min_delta is None else insert_min_delta
        self.candidate_min_delta = (
            min_delta if candidate_min_delta is None else candidate_min_delta
        )
        self.score_mode = score_mode
        self.reassign_active = reassign_active
        self.milp_time_limit = milp_time_limit

    def main(self):
        start = time.time()
        while self.now <= self.T:
            deadline = time.time() + self.step_time_limit
            self.time_pass()
            self.update_heads()

            pool_ids = self.reassignable_request_ids()
            if pool_ids:
                self.assignment_step(pool_ids, deadline=deadline)
                self.regret_repair(deadline=deadline)
                if self.local_iter > 0:
                    self.reinsert_local_search(deadline=deadline)

            self.now += self.dt

        self.ct = time.time() - start
        return self.calculate_ov(mode="posterior"), self.ct

    def assignment_step(self, pool_ids, deadline=None):
        pool_ids = tuple(dict.fromkeys(int(req_id) for req_id in pool_ids))
        if not pool_ids:
            return False

        self.remove_pool_from_future_routes(pool_ids)

        candidates = []
        for k in range(self.fleet_size):
            if deadline is not None and time.time() >= deadline:
                break

            candidates.extend(
                self.generate_vehicle_candidates(k, pool_ids, deadline=deadline)
            )

        selected = self.solve_assignment(candidates, pool_ids, deadline=deadline)
        if not selected:
            self.ov = self.calculate_ov(mode=self.score_mode)
            return False

        for cand in selected:
            self.sol[cand.k] = cand.route.copy()
            for req_id in cand.req_ids:
                self.set_status(req_id, False)

        self.update_heads()
        self.ov = self.calculate_ov(mode=self.score_mode)
        return True

    def reassignable_request_ids(self):
        active = self.data[
            (self.data["pick_early"] <= self.now)
            & (self.data["pick_late"] >= self.now)
        ]
        if len(active) == 0:
            return []

        active_ids = set(int(req_id) for req_id in active["id"])
        if not self.reassign_active:
            return [
                int(req_id)
                for req_id in active["id"]
                if bool(self.data.at[int(req_id), "status"])
            ]

        completed = set()
        for k, route in enumerate(self.sol):
            head = int(self.heads[k])
            if head >= 1:
                completed.update(int(node) for node in route[1 : head + 1, 0])
        return sorted(active_ids - completed)

    def remove_pool_from_future_routes(self, pool_ids):
        pool = set(int(req_id) for req_id in pool_ids)
        for k, route in enumerate(self.sol):
            head = int(self.heads[k])
            keep = np.ones(len(route), dtype=bool)
            for pos in range(head + 1, len(route)):
                if int(route[pos, 0]) in pool:
                    keep[pos] = False

            if not np.all(keep):
                self.sol[k] = self.update_route(
                    route[keep], since=head, mode=self.score_mode
                )

        for req_id in pool:
            self.set_status(req_id, True)

        self.update_heads()
        self.ov = self.calculate_ov(mode=self.score_mode)

    def generate_vehicle_candidates(self, k, pool_ids, deadline=None):
        base_route = self.sol[k]
        head = int(self.heads[k])
        base_value = self.route_objective(
            base_route, since=head, mode=self.score_mode
        )

        singles = []
        for req_id in pool_ids:
            if deadline is not None and time.time() >= deadline:
                break

            req = self.req_map.loc[int(req_id)]
            option = self.best_insert_in_route(k, base_route, req, base_value)
            if option is not None and option["delta"] > self.insert_min_delta:
                singles.append((option["delta"], int(req_id), option))

        if not singles:
            return []

        singles.sort(key=lambda x: x[0], reverse=True)
        req_pool = [req_id for _, req_id, _ in singles[: self.vehicle_pool_cap]]

        candidates_by_key = {}
        beam = [
            {
                "req_set": frozenset(),
                "route": base_route.copy(),
                "value": base_value,
            }
        ]

        for _ in range(self.max_bundle_size):
            expansions_by_key = {}
            for state in beam:
                if deadline is not None and time.time() >= deadline:
                    break

                used = state["req_set"]
                for req_id in req_pool:
                    if deadline is not None and time.time() >= deadline:
                        break

                    if req_id in used:
                        continue

                    req = self.req_map.loc[int(req_id)]
                    option = self.best_insert_in_route(
                        k, state["route"], req, state["value"]
                    )
                    if option is None or option["delta"] <= self.insert_min_delta:
                        continue

                    req_set = frozenset((*used, int(req_id)))
                    key = tuple(sorted(req_set))
                    total_delta = float(option["value"] - base_value)
                    state1 = {
                        "req_set": req_set,
                        "route": option["route"],
                        "value": option["value"],
                        "delta": total_delta,
                    }
                    old = expansions_by_key.get(key)
                    if old is None or state1["delta"] > old["delta"]:
                        expansions_by_key[key] = state1

                    if total_delta > self.candidate_min_delta:
                        cand = TripCandidate(k, key, option["route"], total_delta)
                        old_cand = candidates_by_key.get(key)
                        if old_cand is None or cand.delta > old_cand.delta:
                            candidates_by_key[key] = cand

            if not expansions_by_key:
                break

            beam = sorted(
                expansions_by_key.values(), key=lambda x: x["delta"], reverse=True
            )[: self.beam_width]

        candidates = sorted(
            candidates_by_key.values(), key=lambda x: x.delta, reverse=True
        )
        return candidates[: self.candidate_cap]

    def best_insert_in_route(self, k, route, req, current_value=None):
        req_id = int(req["id"])
        head = int(self.heads[k])
        if current_value is None:
            current_value = self.route_objective(
                route, since=head, mode=self.score_mode
            )

        best = None
        for pos in range(head + 1, len(route) + 1):
            route_base = self.insert_request(route, req_id, 0, pos)
            route_base = self.update_route(
                route_base, since=head, mode=self.score_mode
            )
            for eta in range(3):
                route1 = route_base.copy()
                route1[pos, 1] = eta
                value = self.route_profit(route1, since=head)
                delta = float(value - current_value)
                if best is None or delta > best["delta"]:
                    best = {
                        "delta": delta,
                        "value": value,
                        "k": k,
                        "route": route1,
                    }
        return best

    def insertion_options_scored(self, req, vehi_idx=None, limit=None):
        if vehi_idx is None:
            vehi_idx = range(self.fleet_size)

        options = []
        for k in vehi_idx:
            route = self.sol[k]
            head = int(self.heads[k])
            current_value = self.route_objective(
                route, since=head, mode=self.score_mode
            )
            option = self.best_insert_in_route(k, route, req, current_value)
            if option is not None:
                options.append(option)

        options.sort(key=lambda x: x["delta"], reverse=True)
        if limit is not None:
            return options[:limit]
        return options

    def apply_scored_insertion(self, best):
        head = int(self.heads[best["k"]])
        self.sol[best["k"]] = self.update_route(
            best["route"], since=head, mode=self.score_mode
        )
        self.ov += best["delta"]

    def solve_assignment(self, candidates, pool_ids, deadline=None):
        candidates = [cand for cand in candidates if cand.delta > 0]
        if not candidates:
            return []

        if milp is None or sparse is None:
            return self.greedy_assignment(candidates)

        req_to_row = {int(req_id): i for i, req_id in enumerate(pool_ids)}
        n_vars = len(candidates)
        n_rows = self.fleet_size + len(req_to_row)
        row_idx = []
        col_idx = []
        data = []

        for j, cand in enumerate(candidates):
            row_idx.append(cand.k)
            col_idx.append(j)
            data.append(1.0)
            for req_id in cand.req_ids:
                row = req_to_row.get(int(req_id))
                if row is not None:
                    row_idx.append(self.fleet_size + row)
                    col_idx.append(j)
                    data.append(1.0)

        A = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(n_rows, n_vars))
        constraints = LinearConstraint(A.tocsr(), -np.inf, np.ones(n_rows))
        c = -np.array([cand.delta for cand in candidates], dtype=float)

        try:
            time_limit = self.milp_time_limit
            if deadline is not None:
                time_limit = max(0.0, min(time_limit, deadline - time.time()))
            if time_limit <= 0:
                return self.greedy_assignment(candidates)

            result = milp(
                c=c,
                integrality=np.ones(n_vars),
                bounds=Bounds(0, 1),
                constraints=constraints,
                options={"time_limit": time_limit, "disp": False},
            )
        except Exception:
            return self.greedy_assignment(candidates)

        if result.x is None:
            return self.greedy_assignment(candidates)

        selected = [
            candidates[i]
            for i, value in enumerate(result.x)
            if value is not None and value > 0.5
        ]
        if not selected:
            return self.greedy_assignment(candidates)
        return selected

    @staticmethod
    def greedy_assignment(candidates):
        selected = []
        used_vehicles = set()
        used_requests = set()

        for cand in sorted(candidates, key=lambda x: x.delta, reverse=True):
            if cand.k in used_vehicles:
                continue
            req_set = set(cand.req_ids)
            if used_requests.intersection(req_set):
                continue
            selected.append(cand)
            used_vehicles.add(cand.k)
            used_requests.update(req_set)

        return selected

    def regret_repair(self, deadline=None):
        for _ in range(self.repair_rounds):
            if deadline is not None and time.time() >= deadline:
                break

            improved = False
            while True:
                if deadline is not None and time.time() >= deadline:
                    break

                active = self.get_active_requests(sort_fare=False)
                best_choice = None

                for _, req in active.iterrows():
                    req_id = int(req["id"])
                    if not bool(self.data.at[req_id, "status"]):
                        continue

                    options = self.insertion_options_scored(req, limit=2)
                    if not options or options[0]["delta"] <= self.min_delta:
                        continue

                    second = options[1]["delta"] if len(options) > 1 else self.min_delta
                    regret = options[0]["delta"] - second
                    score = regret + 0.05 * options[0]["delta"]
                    if best_choice is None or score > best_choice["score"]:
                        best_choice = {
                            "score": score,
                            "req_id": req_id,
                            "best": options[0],
                        }

                if best_choice is None:
                    break

                self.apply_scored_insertion(best_choice["best"])
                self.set_status(best_choice["req_id"], False)
                improved = True

            if not improved:
                break

        self.ov = self.calculate_ov(mode=self.score_mode)

    def removal_candidates_scored(self, active_only=True):
        cands = []
        for k, route in enumerate(self.sol):
            head = int(self.heads[k])
            if len(route) <= head + 1:
                continue

            base_value = self.route_objective(
                route, since=head, mode=self.score_mode
            )
            for pos in range(head + 1, len(route)):
                req_id = int(route[pos, 0])
                req = self.reqs[req_id]
                if active_only and float(req["pick_late"]) < self.now:
                    continue

                route1 = np.delete(route, pos, axis=0)
                route1 = self.update_route(route1, since=head, mode=self.score_mode)
                value = self.route_profit(route1, since=head)
                cands.append((float(value - base_value), req_id, k, pos))
        return cands

    def remove_request_scored(self, req_id):
        loc = self.find_request(req_id)
        if loc is None:
            return False, 0.0

        k, pos = loc
        head = int(self.heads[k])
        route = self.sol[k]
        base_value = self.route_objective(route, since=head, mode=self.score_mode)
        route1 = np.delete(route, pos, axis=0)
        route1 = self.update_route(route1, since=head, mode=self.score_mode)
        value = self.route_profit(route1, since=head)
        delta = float(value - base_value)

        self.sol[k] = route1
        self.ov += delta
        self.set_status(req_id, True)
        return True, delta

    def reinsert_local_search(self, deadline=None):
        improved = True
        n_iter = 0

        while improved and n_iter < self.local_iter:
            if deadline is not None and time.time() >= deadline:
                break

            improved = False
            n_iter += 1
            best_move = None
            cands = self.removal_candidates_scored(active_only=True)
            cands.sort(key=lambda x: x[0], reverse=True)

            for _, req_id, _, _ in cands[: self.local_candidate_cap]:
                if deadline is not None and time.time() >= deadline:
                    break

                state = self.snapshot()
                ok, remove_delta = self.remove_request_scored(req_id)
                if not ok:
                    self.restore(state)
                    continue

                req = self.req_map.loc[int(req_id)]
                options = self.insertion_options_scored(req, limit=1)
                if options:
                    total_delta = remove_delta + options[0]["delta"]
                    if best_move is None or total_delta > best_move["delta"]:
                        best_move = {
                            "delta": total_delta,
                            "state": state,
                            "req_id": int(req_id),
                            "insert": options[0],
                        }
                self.restore(state)

            if best_move is not None and best_move["delta"] > 1e-9:
                self.restore(best_move["state"])
                self.remove_request_scored(best_move["req_id"])
                req = self.req_map.loc[best_move["req_id"]]
                insert = self.insertion_options_scored(req, limit=1)[0]
                self.apply_scored_insertion(insert)
                self.set_status(best_move["req_id"], False)
                improved = True

        self.ov = self.calculate_ov(mode=self.score_mode)


def main(
    inst_path=DEFAULT_INST_PATH,
    steps=STEPS,
    top_n=None,
    seed=None,
    max_bundle_size=4,
    beam_width=14,
    vehicle_pool_cap=22,
    candidate_cap=45,
    repair_rounds=1,
    local_iter=0,
    min_delta=5.0,
    score_mode="prior",
    fleet_size=K,
    step_time_limit=300.0,
):
    ins = Instance(inst_path, top_n=top_n)
    bta = BatchTripAssignment(
        ins.data,
        steps=steps,
        seed=seed,
        max_bundle_size=max_bundle_size,
        beam_width=beam_width,
        vehicle_pool_cap=vehicle_pool_cap,
        candidate_cap=candidate_cap,
        repair_rounds=repair_rounds,
        local_iter=local_iter,
        min_delta=min_delta,
        score_mode=score_mode,
        fleet_size=fleet_size,
        step_time_limit=step_time_limit,
    )
    ov, ct = bta.main()
    print(f"Objective value: {ov:.3f} | Runtime: {ct:.3f} sec.")
    return ov, ct


INST_DIR = Path("Data/Test/XS-size")
INSTS = [
    "XS_a01.parquet",
]

FLEET_SIZE = 20
if __name__ == "__main__":
    for ins in INSTS:
        main(inst_path=INST_DIR / ins, fleet_size=FLEET_SIZE)
