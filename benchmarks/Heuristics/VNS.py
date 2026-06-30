# -*- coding: utf-8 -*-
"""A compact VNS benchmark for the rolling-horizon ridesharing problem."""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np

try:
    from .GI import DEFAULT_INST_PATH, K, STEPS, Instance, HeuristicRollingHorizon
except ImportError:
    from GI import DEFAULT_INST_PATH, K, STEPS, Instance, HeuristicRollingHorizon


class VNS(HeuristicRollingHorizon):
    def __init__(
        self,
        data,
        steps=STEPS,
        seed=42,
        tolerance=3,
        max_local_iter=1,
        candidate_cap=8,
        base_shake_rate=0.15,
        shake_temp=2.0,
        shake_temp_max=10.0,
        shake_gamma=2.0,
        selection_mode="posterior",
        min_delta=10.0,
        fleet_size=K,
        step_time_limit=300.0,
        repair_candidate_cap=80,
        repair_vehicle_cap=20,
    ):
        super().__init__(
            data,
            steps=steps,
            seed=seed,
            min_delta=min_delta,
            fleet_size=fleet_size,
            step_time_limit=step_time_limit,
        )
        self.tolerance = tolerance
        self.max_local_iter = max_local_iter
        self.candidate_cap = candidate_cap
        self.base_shake_rate = base_shake_rate
        self.shake_temp = shake_temp
        self.shake_temp_max = shake_temp_max
        self.shake_gamma = shake_gamma
        self.selection_mode = selection_mode
        self.step_time_limit = step_time_limit
        self.repair_candidate_cap = repair_candidate_cap
        self.repair_vehicle_cap = repair_vehicle_cap

    def main(self):
        start = time.time()
        while self.now <= self.T:
            deadline = time.time() + self.step_time_limit
            self.time_pass()
            self.update_heads()
            active = self.get_active_requests()
            if len(active) > 0:
                improved, _ = self.greedy_insert(active, deadline=deadline)
                if improved:
                    self.vns_process(deadline=deadline)
            self.now += self.dt

        self.ct = time.time() - start
        return self.calculate_ov(mode="posterior"), self.ct

    def vns_process(self, deadline=None):
        best_state = self.snapshot()
        best_score = self.state_score()
        fail = 0

        while fail < self.tolerance:
            if deadline is not None and time.time() >= deadline:
                break

            self.restore(best_state)
            shake_rate = min(0.60, self.base_shake_rate + 0.05 * fail)
            self.shake(shake_rate=shake_rate, temp=self.shake_temperature(fail))
            self.regret_insert(deadline=deadline)
            if self.max_local_iter > 0:
                self.reinsert_local_search(deadline=deadline)

            score = self.state_score()
            if score > best_score + 1e-9:
                best_score = score
                best_state = self.snapshot()
                fail = 0
            else:
                fail += 1

        self.restore(best_state)

    def state_score(self):
        if self.selection_mode == "prior":
            return self.ov
        return self.calculate_ov(mode=self.selection_mode)

    def shake_temperature(self, fail):
        if self.tolerance <= 1:
            return self.shake_temp
        x = fail / max(self.tolerance - 1, 1)
        return self.shake_temp + (self.shake_temp_max - self.shake_temp) * x**self.shake_gamma

    def shake(self, shake_rate=0.20, temp=2.0):
        cands = self.removal_candidates(active_only=True)
        if not cands:
            return []

        deltas = np.array([delta for delta, _, _, _ in cands], dtype=float)
        req_ids = np.array([req_id for _, req_id, _, _ in cands], dtype=int)
        n_remove = max(1, int(np.ceil(len(req_ids) * shake_rate)))
        n_remove = min(n_remove, len(req_ids))

        z = (deltas - deltas.max()) / max(temp, 1e-9)
        probs = np.exp(z)
        probs = probs / probs.sum()
        removed_ids = self.rng.choice(
            req_ids, size=n_remove, replace=False, p=probs
        )

        removed = []
        for req_id in removed_ids:
            ok, _ = self.remove_request(int(req_id))
            if ok:
                removed.append(int(req_id))
        return removed

    def regret_insert(self, regret_k=3, deadline=None):
        improved = False

        while True:
            if deadline is not None and time.time() >= deadline:
                break

            active = self.get_active_requests(sort_fare=False)
            if self.repair_candidate_cap is not None and len(active) > self.repair_candidate_cap:
                active = active.sort_values(by="fare", ascending=False).head(
                    self.repair_candidate_cap
                )
            best_choice = None

            for _, req in active.iterrows():
                req_id = int(req["id"])
                if not bool(self.data.at[req_id, "status"]):
                    continue

                vehi_idx = self.vehicle_candidates(req, self.repair_vehicle_cap)
                options = self.insertion_options(
                    req, vehi_idx=vehi_idx, limit=regret_k, deadline=deadline
                )
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

            self.apply_insertion(best_choice["best"])
            self.set_status(best_choice["req_id"], False)
            improved = True

        return improved

    def vehicle_candidates(self, req, cap=None):
        if cap is None or cap >= self.fleet_size:
            return range(self.fleet_size)

        pick = np.array([req["pick_longt"], req["pick_lat"]])
        scored = []
        for k, route in enumerate(self.sol):
            head = int(self.heads[k])
            node_id = int(route[head, 0])
            coord = self.get_node_coord(node_id)
            pending = max(0, len(route) - head - 1)
            score = self.get_distance(coord, pick) + 0.05 * pending
            scored.append((score, k))

        scored.sort(key=lambda x: x[0])
        return [k for _, k in scored[: max(1, int(cap))]]

    def reinsert_local_search(self, deadline=None):
        improved = True
        n_iter = 0

        while improved and n_iter < self.max_local_iter:
            if deadline is not None and time.time() >= deadline:
                break

            improved = False
            n_iter += 1
            best_move = None
            cands = self.removal_candidates(active_only=True)
            cands.sort(key=lambda x: x[0], reverse=True)

            for _, req_id, _, _ in cands[: self.candidate_cap]:
                if deadline is not None and time.time() >= deadline:
                    break

                state = self.snapshot()
                ok, remove_delta = self.remove_request(req_id)
                if not ok:
                    self.restore(state)
                    continue

                req = self.req_map.loc[int(req_id)]
                vehi_idx = self.vehicle_candidates(req, self.repair_vehicle_cap)
                insert = self.best_insert(req, vehi_idx=vehi_idx, deadline=deadline)
                if insert is not None:
                    total_delta = remove_delta + insert["delta"]
                    if best_move is None or total_delta > best_move["delta"]:
                        best_move = {
                            "delta": total_delta,
                            "state": state,
                            "req_id": int(req_id),
                        }
                self.restore(state)

            if best_move is not None and best_move["delta"] > 1e-9:
                self.restore(best_move["state"])
                self.remove_request(best_move["req_id"])
                req = self.req_map.loc[best_move["req_id"]]
                vehi_idx = self.vehicle_candidates(req, self.repair_vehicle_cap)
                insert = self.best_insert(req, vehi_idx=vehi_idx, deadline=deadline)
                if insert is not None:
                    self.apply_insertion(insert)
                    self.set_status(best_move["req_id"], False)
                    improved = True


def main(
    inst_path=Path("Data/Test/XS-size/XS_a01.parquet"),
    steps=STEPS,
    top_n=None,
    seed=None,
    tolerance=3,
    max_local_iter=1,
    candidate_cap=8,
    min_delta=10.0,
    shake_temp=2.0,
    shake_temp_max=10.0,
    shake_gamma=2.0,
    selection_mode="posterior",
    fleet_size=20,
    step_time_limit=300.0,
    repair_candidate_cap=80,
    repair_vehicle_cap=20,
):
    ins = Instance(inst_path, top_n=top_n)
    vns = VNS(
        ins.data,
        steps=steps,
        seed=seed,
        tolerance=tolerance,
        max_local_iter=max_local_iter,
        candidate_cap=candidate_cap,
        min_delta=min_delta,
        shake_temp=shake_temp,
        shake_temp_max=shake_temp_max,
        shake_gamma=shake_gamma,
        selection_mode=selection_mode,
        fleet_size=fleet_size,
        step_time_limit=step_time_limit,
        repair_candidate_cap=repair_candidate_cap,
        repair_vehicle_cap=repair_vehicle_cap,
    )
    ov, ct = vns.main()
    print(f"Objective value: {ov:.3f} | Runtime: {ct:.3f} sec.")
    return ov, ct


INST_DIR = Path('Data/Test/XS-size')
INSTS = ['XS_a01.parquet']

if __name__ == "__main__":
    for ins in INSTS:
        main(inst_path=INST_DIR / ins)
