# -*- coding: utf-8 -*-
"""A compact ALNS benchmark for the rolling-horizon ridesharing problem."""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np

try:
    from .GI import DEFAULT_INST_PATH, K, STEPS, Instance, HeuristicRollingHorizon
except ImportError:
    from GI import DEFAULT_INST_PATH, K, STEPS, Instance, HeuristicRollingHorizon


class ALNS(HeuristicRollingHorizon):
    def __init__(
        self,
        data,
        steps=STEPS,
        seed=None,
        iterations=5,
        destroy_rate=0.10,
        temperature=5.0,
        cooling=0.90,
        min_delta=12.0,
        fleet_size=K,
        step_time_limit=300.0,
        selection_mode="posterior",
        regret_k=3,
        local_iter=1,
        candidate_cap=8,
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
        self.iterations = iterations
        self.destroy_rate = destroy_rate
        self.temperature = temperature
        self.cooling = cooling
        self.step_time_limit = step_time_limit
        self.selection_mode = selection_mode
        self.regret_k = regret_k
        self.local_iter = local_iter
        self.candidate_cap = candidate_cap
        self.repair_candidate_cap = repair_candidate_cap
        self.repair_vehicle_cap = repair_vehicle_cap
        self.destroy_ops = ("worst", "random", "related")
        self.destroy_weights = np.ones(len(self.destroy_ops), dtype=float)

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
                    self.alns_process(deadline=deadline)
            self.now += self.dt

        self.ct = time.time() - start
        return self.calculate_ov(mode="posterior"), self.ct

    def alns_process(self, deadline=None):
        best_state = self.snapshot()
        best_score = self.state_score()
        temp = self.temperature

        for _ in range(self.iterations):
            if deadline is not None and time.time() >= deadline:
                break

            old_state = self.snapshot()
            old_score = self.state_score()

            op_idx = self.rng.choice(
                len(self.destroy_ops),
                p=self.destroy_weights / self.destroy_weights.sum(),
            )
            removed = self.destroy(self.destroy_ops[op_idx])
            if removed:
                self.regret_insert(regret_k=self.regret_k, deadline=deadline)
                if self.local_iter > 0:
                    self.reinsert_local_search(deadline=deadline)

            score = self.state_score()
            delta = score - old_score
            accept = delta >= 0 or self.rng.random() < np.exp(delta / max(temp, 1e-9))
            if accept:
                if score > best_score + 1e-9:
                    best_score = score
                    best_state = self.snapshot()
                    self.destroy_weights[op_idx] += 3.0
                else:
                    self.destroy_weights[op_idx] += 0.5
            else:
                self.restore(old_state)

            temp *= self.cooling

        self.restore(best_state)

    def state_score(self):
        if self.selection_mode == "prior":
            return float(self.ov)
        return self.calculate_ov(mode=self.selection_mode)

    def destroy(self, name):
        cands = self.removal_candidates(active_only=True)
        if not cands:
            return []

        n_remove = max(1, int(np.ceil(len(cands) * self.destroy_rate)))
        if name == "worst":
            cands.sort(key=lambda x: x[0], reverse=True)
            selected = cands[:n_remove]
        elif name == "random":
            idx = self.rng.choice(len(cands), size=min(n_remove, len(cands)), replace=False)
            selected = [cands[i] for i in idx]
        else:
            selected = self.related_candidates(cands, n_remove)

        removed = []
        for _, req_id, _, _ in selected:
            ok, _ = self.remove_request(req_id)
            if ok:
                removed.append(req_id)
        return removed

    def related_candidates(self, cands, n_remove):
        seed = cands[int(self.rng.integers(len(cands)))]
        seed_req = self.reqs[int(seed[1])]
        seed_pick = np.array([seed_req["pick_longt"], seed_req["pick_lat"]])
        seed_time = float(seed_req["pick_early"])

        scored = []
        for cand in cands:
            req = self.reqs[int(cand[1])]
            pick = np.array([req["pick_longt"], req["pick_lat"]])
            dis = self.get_distance(seed_pick, pick)
            time_gap = abs(float(req["pick_early"]) - seed_time)
            fare_gap = abs(float(req["fare"]) - float(seed_req["fare"])) / 10
            scored.append((dis + time_gap + fare_gap, cand))
        scored.sort(key=lambda x: x[0])
        return [cand for _, cand in scored[:n_remove]]

    def regret_insert(self, regret_k=3, deadline=None):
        improved = False
        total_delta = 0.0

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
                    best_choice = {"score": score, "req_id": req_id, "best": options[0]}

            if best_choice is None:
                break

            self.apply_insertion(best_choice["best"])
            self.set_status(best_choice["req_id"], False)
            total_delta += best_choice["best"]["delta"]
            improved = True

        return improved, total_delta

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

        while improved and n_iter < self.local_iter:
            if deadline is not None and time.time() >= deadline:
                break

            improved = False
            n_iter += 1
            base_score = self.state_score()
            best_move = None
            cands = self.removal_candidates(active_only=True)
            cands.sort(key=lambda x: x[0], reverse=True)

            for _, req_id, _, _ in cands[: self.candidate_cap]:
                if deadline is not None and time.time() >= deadline:
                    break

                state = self.snapshot()
                ok, _ = self.remove_request(req_id)
                if not ok:
                    self.restore(state)
                    continue

                req = self.req_map.loc[int(req_id)]
                vehi_idx = self.vehicle_candidates(req, self.repair_vehicle_cap)
                insert = self.best_insert(req, vehi_idx=vehi_idx, deadline=deadline)
                if insert is not None and insert["delta"] > self.min_delta:
                    self.apply_insertion(insert)
                    self.set_status(req_id, False)
                    score = self.state_score()
                    if score > base_score + 1e-9 and (
                        best_move is None or score > best_move["score"]
                    ):
                        best_move = {
                            "score": score,
                            "state": state,
                            "req_id": int(req_id),
                        }
                self.restore(state)

            if best_move is not None:
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
    iterations=10,
    destroy_rate=0.10,
    min_delta=5.0,
    fleet_size=100,
    step_time_limit=300.0,
    selection_mode="posterior",
    regret_k=3,
    local_iter=1,
    candidate_cap=8,
    repair_candidate_cap=80,
    repair_vehicle_cap=20,
):
    ins = Instance(inst_path, top_n=top_n)
    alns = ALNS(
        ins.data,
        steps=steps,
        seed=seed,
        iterations=iterations,
        destroy_rate=destroy_rate,
        min_delta=min_delta,
        fleet_size=fleet_size,
        step_time_limit=step_time_limit,
        selection_mode=selection_mode,
        regret_k=regret_k,
        local_iter=local_iter,
        candidate_cap=candidate_cap,
        repair_candidate_cap=repair_candidate_cap,
        repair_vehicle_cap=repair_vehicle_cap,
    )
    ov, ct = alns.main()
    print(f"Objective value: {ov:.3f} | Runtime: {ct:.3f} sec.")
    return ov, ct

INST_DIR = Path('Data/Test/XS-size')
INSTS = ['XS_a01.parquet']

if __name__ == "__main__":
    for ins in INSTS:
        main(inst_path=INST_DIR / ins)
