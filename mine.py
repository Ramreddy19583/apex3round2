from __future__ import annotations

import math
import sys
import time
from math import isqrt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter import Router, assign_vehicles, neighbour_lists, route_cost, simulate_route  # noqa: E402
from benchkit.rng import Rng  # noqa: E402
from data import Instance  # noqa: E402

SUBMIT_INTERVAL = 0.05  # seconds between submissions while improving
RESERVE = 0.15  # stop this far from the deadline so the final submit lands


def _null_sink(_plan) -> dict:
    return {
        "accepted": False,
        "reason": "no sink",
        "cost": None,
        "best": None,
        "elapsed_s": 0.0,
        "remaining_s": None,
    }


class EnhancedSavingsRouter(Router):
    def prepare(self, instance: Instance, deadline: float) -> None:
        self.inst = instance
        self.n = instance.n
        self.started = time.perf_counter()
        self.deadline = deadline
        self.last_submit = 0.0
        self.best_cost: Optional[int] = None
        self.best_plan: Optional[List[List[int]]] = None
        self.rng = Rng(instance.digest() ^ 0xC1A2_4E00)
        self.neighbours = neighbour_lists(instance, k=min(32, max(2, instance.n - 1)))
        self.submit = _null_sink
        self._precalculate_shaw_matrix()

    def _precalculate_shaw_matrix(self) -> None:
        inst = self.inst
        n = self.n
        self.shaw_dist = [[0.0] * (n + 1) for _ in range(n + 1)]
        max_dist = 1
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                d = inst.travel(i, j)
                if d > max_dist:
                    max_dist = d

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i == j:
                    continue
                d = inst.travel(i, j) / max_dist
                tw_diff = abs(inst.tw_open[i] - inst.tw_open[j]) / max(1, inst.horizon)
                cap_diff = abs(inst.demand[i] - inst.demand[j]) / max(1, max(inst.capacities))
                self.shaw_dist[i][j] = 0.5 * d + 0.3 * tw_diff + 0.2 * cap_diff

    def initial_plan(self) -> Optional[List[List[int]]]:
        plan = self._regret_insertion_construct()
        if plan is None:
            plan = self._greedy_append()
        if plan is None:
            plan = self._sweep_construct()
        if plan is None:
            return None

        best, best_cost = plan, self._plan_cost(plan)
        merged = self._clarke_wright()
        if merged is not None:
            cost = self._plan_cost(merged)
            if cost is not None and (best_cost is None or cost < best_cost):
                best, best_cost = merged, cost
        return best

    def solve(self, instance: Instance, submit_candidate) -> object:
        self.prepare(instance, time.perf_counter() + 1e9)
        self.submit = submit_candidate

        plan = self.initial_plan()
        if plan is None:
            return []
        self._offer(plan)

        self._alns_local_search()
        return self.best_plan if self.best_plan is not None else plan

    # -- Bookkeeping --------------------------------------------------------

    def _elapsed(self) -> float:
        return time.perf_counter() - self.started

    def _remaining(self) -> float:
        return self.deadline - time.perf_counter()

    def _offer(self, plan: Sequence[Sequence[int]], force: bool = True) -> bool:
        now = self._elapsed()
        if not force and now - self.last_submit < SUBMIT_INTERVAL:
            return False
        receipt = self.submit([list(route) for route in plan])
        self.last_submit = now
        if receipt.get("remaining_s") is not None:
            self.deadline = time.perf_counter() + receipt["remaining_s"]
        if receipt["accepted"]:
            cost = receipt["cost"]
            if self.best_cost is None or cost < self.best_cost:
                self.best_cost = cost
                self.best_plan = [list(route) for route in plan]
            return True
        return False

    def _plan_cost(self, plan: Sequence[Sequence[int]]) -> Optional[int]:
        total = 0
        for vehicle, route in enumerate(plan):
            if not route:
                continue
            value = route_cost(self.inst, route, vehicle)
            if value is None:
                return None
            total += value
        return total

    # -- Regret-k Construction ----------------------------------------------

    def _regret_insertion_construct(self, k_regret: int = 3) -> Optional[List[List[int]]]:
        inst = self.inst
        plan: List[List[int]] = [[] for _ in range(inst.num_vehicles)]
        loads = [0] * inst.num_vehicles
        unassigned = set(range(1, self.n + 1))

        while unassigned and self._remaining() > RESERVE:
            best_customer = -1
            max_regret = -1.0
            best_insertion = None

            for customer in unassigned:
                options = []
                for vehicle in range(inst.num_vehicles):
                    if loads[vehicle] + inst.demand[customer] > inst.capacities[vehicle]:
                        continue
                    route = plan[vehicle]
                    for pos in range(len(route) + 1):
                        trial = route[:pos] + [customer] + route[pos:]
                        if simulate_route(inst, trial, vehicle) is not None:
                            old_c = route_cost(inst, route, vehicle) or 0
                            new_c = route_cost(inst, trial, vehicle) or 0
                            options.append((new_c - old_c, vehicle, pos))

                if not options:
                    return None

                options.sort(key=lambda x: x[0])
                cost_best = options[0][0]
                cost_k = options[min(k_regret - 1, len(options) - 1)][0]
                regret = cost_k - cost_best

                if regret > max_regret or best_customer == -1:
                    max_regret = regret
                    best_customer = customer
                    best_insertion = (options[0][1], options[0][2])

            if best_customer == -1 or best_insertion is None:
                return None

            v, pos = best_insertion
            plan[v].insert(pos, best_customer)
            loads[v] += inst.demand[best_customer]
            unassigned.remove(best_customer)

        return None if unassigned else plan

    # -- Fallback Constructions ---------------------------------------------

    def _greedy_append(self) -> Optional[List[List[int]]]:
        inst = self.inst
        orders = (
            sorted(range(1, self.n + 1), key=lambda c: (inst.tw_close[c], inst.tw_open[c], c)),
            sorted(range(1, self.n + 1), key=lambda c: (inst.tw_open[c], inst.tw_close[c], c)),
        )
        best: Optional[List[List[int]]] = None
        best_cost: Optional[int] = None
        for order in orders:
            plan = self._append_pass(order)
            if plan is None:
                continue
            cost = self._plan_cost(plan)
            if cost is not None and (best_cost is None or cost < best_cost):
                best, best_cost = plan, cost
        return best

    def _append_pass(self, order: Sequence[int]) -> Optional[List[List[int]]]:
        inst = self.inst
        xs, ys = inst.xs, inst.ys
        vehicles = inst.num_vehicles

        routes: List[List[int]] = [[] for _ in range(vehicles)]
        clock = list(inst.starts)
        last = [0] * vehicles
        load = [0] * vehicles

        for customer in order:
            demand = inst.demand[customer]
            open_at, close_at = inst.tw_open[customer], inst.tw_close[customer]
            service = inst.service[customer]
            cx, cy = xs[customer], ys[customer]
            home = isqrt((cx - xs[0]) ** 2 + (cy - ys[0]) ** 2)

            chosen = -1
            chosen_cost = None
            chosen_clock = 0
            for v in range(vehicles):
                if load[v] + demand > inst.capacities[v]:
                    continue
                prev = last[v]
                step = isqrt((xs[prev] - cx) ** 2 + (ys[prev] - cy) ** 2)
                arrival = clock[v] + step
                if arrival > close_at:
                    continue
                departure = (open_at if arrival < open_at else arrival) + service
                if departure + home - inst.starts[v] > inst.max_duration:
                    continue
                if departure + home > inst.horizon:
                    continue
                back = isqrt((xs[prev] - xs[0]) ** 2 + (ys[prev] - ys[0]) ** 2)
                marginal = step + home - back
                if chosen_cost is None or marginal < chosen_cost:
                    chosen, chosen_cost, chosen_clock = v, marginal, departure

            if chosen < 0:
                return None
            routes[chosen].append(customer)
            clock[chosen] = chosen_clock
            last[chosen] = customer
            load[chosen] += demand

        return routes

    def _sweep_construct(self) -> Optional[List[List[int]]]:
        inst = self.inst
        depot_x, depot_y = inst.xs[0], inst.ys[0]

        def angle(customer: int) -> int:
            dx, dy = inst.xs[customer] - depot_x, inst.ys[customer] - depot_y
            total = abs(dx) + abs(dy)
            if total == 0:
                return 0
            if dx >= 0 and dy >= 0:
                return 10000 * dy // total
            if dx < 0 and dy >= 0:
                return 10000 + 10000 * (-dx) // total
            if dx < 0 and dy < 0:
                return 20000 + 10000 * (-dy) // total
            return 30000 + 10000 * dx // total

        order = sorted(range(1, self.n + 1), key=lambda c: (angle(c), c))
        vehicles = sorted(
            range(inst.num_vehicles),
            key=lambda v: (inst.starts[v], -inst.capacities[v], v),
        )

        plan: List[List[int]] = [[] for _ in range(inst.num_vehicles)]
        pending = list(order)
        for vehicle in vehicles:
            if not pending:
                break
            if self._remaining() < RESERVE:
                return None
            capacity = inst.capacities[vehicle]
            wedge: List[int] = []
            load = 0
            while pending:
                taken = -1
                for offset in range(min(12, len(pending))):
                    customer = pending[offset]
                    if load + inst.demand[customer] > capacity:
                        continue
                    positions = []
                    for position in range(len(wedge) + 1):
                        previous = wedge[position - 1] if position else 0
                        following = wedge[position] if position < len(wedge) else 0
                        positions.append(
                            (
                                inst.travel(previous, customer)
                                + inst.travel(customer, following)
                                - inst.travel(previous, following),
                                position,
                            )
                        )
                    positions.sort()
                    for _, position in positions[:4]:
                        candidate = wedge[:position] + [customer] + wedge[position:]
                        if simulate_route(inst, candidate, vehicle) is not None:
                            wedge = candidate
                            load += inst.demand[customer]
                            taken = offset
                            break
                    if taken >= 0:
                        break
                if taken < 0:
                    break
                pending.pop(taken)
            plan[vehicle] = wedge

        return None if pending else plan

    def _clarke_wright(self) -> Optional[List[List[int]]]:
        inst = self.inst
        proxy = max(
            range(inst.num_vehicles),
            key=lambda v: (inst.starts[v] == 0, inst.capacities[v]),
        )
        cap_bound = inst.capacities[proxy]

        routes: dict = {}
        route_of: dict = {}
        for customer in range(1, self.n + 1):
            if simulate_route(inst, [customer], proxy) is None:
                return None
            routes[customer] = [customer]
            route_of[customer] = customer

        savings = []
        for i in range(1, self.n + 1):
            d0i = inst.travel(0, i)
            for j in self.neighbours[i]:
                if j <= 0 or j == i:
                    continue
                value = d0i + inst.travel(0, j) - inst.travel(i, j)
                if value > 0:
                    savings.append((-value, i, j))
        savings.sort()

        loads = {c: inst.demand[c] for c in range(1, self.n + 1)}
        for _, i, j in savings:
            if self._remaining() < RESERVE:
                break
            a, b = route_of[i], route_of[j]
            if a == b:
                continue
            if routes[a][-1] != i or routes[b][0] != j:
                continue
            if loads[a] + loads[b] > cap_bound:
                continue
            merged = routes[a] + routes[b]
            if simulate_route(inst, merged, proxy) is None:
                continue
            routes[a] = merged
            loads[a] += loads[b]
            for customer in routes[b]:
                route_of[customer] = a
            del routes[b], loads[b]

        active = [route for route in routes.values() if route]
        if len(active) > inst.num_vehicles:
            return None
        return assign_vehicles(inst, active)

    # -- Adaptive Large Neighborhood Search (ALNS) + Local Search -----------

    def _alns_local_search(self) -> None:
        if self.best_plan is None:
            return

        plan = [list(r) for r in self.best_plan]
        costs = [route_cost(self.inst, r, v) or 0 for v, r in enumerate(plan)]
        where = {c: v for v, r in enumerate(plan) for c in r}

        self._fast_descend(plan, costs, where)
        self._offer(plan)
        current_cost = sum(costs)

        temperature = current_cost * 0.05
        cooling_rate = 0.995

        while self._remaining() > RESERVE:
            trial = [list(r) for r in plan]
            trial_costs = list(costs)
            trial_where = dict(where)

            # Adaptive Ruin Strategy
            ruin_type = self.rng.below(3)
            if ruin_type == 0:
                removed = self._shaw_ruin(trial, trial_where)
            elif ruin_type == 1:
                removed = self._worst_cost_ruin(trial, trial_costs, trial_where)
            else:
                removed = self._cluster_ruin(trial, trial_where)

            if not removed:
                continue

            # Update costs after removal
            for v in range(len(trial)):
                trial_costs[v] = route_cost(self.inst, trial[v], v) or 0

            # Repair using Regret Insertion
            if not self._repair_regret(trial, trial_costs, trial_where, removed):
                continue

            # Fast Local Search Hill Climbing
            self._fast_descend(trial, trial_costs, trial_where)
            trial_total = sum(trial_costs)

            delta = trial_total - current_cost
            if delta < 0 or (temperature > 1e-3 and self.rng.uniform(0, 1) < math.exp(-delta / temperature)):
                plan, costs, where, current_cost = trial, trial_costs, trial_where, trial_total
                self._offer(plan, force=False)

            temperature *= cooling_rate

        self._offer(plan)

    def _shaw_ruin(self, plan: List[List[int]], where: Dict[int, int]) -> List[int]:
        total_customers = self.n
        num_remove = max(4, min(int(total_customers * 0.25), 35))
        removed = []

        seed = self.rng.randint(1, total_customers)
        removed.append(seed)

        while len(removed) < num_remove:
            target = self.rng.choice(removed)
            candidates = sorted(
                [c for c in range(1, total_customers + 1) if c not in removed and c in where],
                key=lambda c: self.shaw_dist[target][c],
            )
            if not candidates:
                break
            idx = int(self.rng.uniform(0, 1) ** 3 * len(candidates[:10]))
            idx = min(idx, len(candidates) - 1)
            removed.append(candidates[idx])

        for c in removed:
            v = where[c]
            plan[v].remove(c)
            del where[c]

        return removed

    def _worst_cost_ruin(self, plan: List[List[int]], costs: List[int], where: Dict[int, int]) -> List[int]:
        num_remove = max(4, min(int(self.n * 0.20), 30))
        savings = []

        for c in range(1, self.n + 1):
            if c not in where:
                continue
            v = where[c]
            route = plan[v]
            if len(route) <= 1:
                savings.append((costs[v], c))
                continue
            idx = route.index(c)
            prev_node = route[idx - 1] if idx > 0 else 0
            next_node = route[idx + 1] if idx < len(route) - 1 else 0
            cost_with = self.inst.travel(prev_node, c) + self.inst.travel(c, next_node)
            cost_without = self.inst.travel(prev_node, next_node)
            savings.append((cost_with - cost_without, c))

        savings.sort(reverse=True)
        removed = [c for _, c in savings[:num_remove]]

        for c in removed:
            v = where[c]
            plan[v].remove(c)
            del where[c]

        return removed

    def _cluster_ruin(self, plan: List[List[int]], where: Dict[int, int]) -> List[int]:
        num_remove = max(4, min(int(self.n * 0.25), 35))
        center = self.rng.randint(1, self.n)
        removed = sorted(
            [c for c in range(1, self.n + 1) if c in where],
            key=lambda c: self.inst.travel(center, c),
        )[:num_remove]

        for c in removed:
            v = where[c]
            plan[v].remove(c)
            del where[c]

        return removed

    def _repair_regret(
        self,
        plan: List[List[int]],
        costs: List[int],
        where: Dict[int, int],
        removed: List[int],
    ) -> bool:
        inst = self.inst
        loads = [sum(inst.demand[c] for c in r) for r in plan]

        while removed:
            best_customer = -1
            max_regret = -1.0
            best_insertion = None

            for customer in removed:
                options = []
                for vehicle in range(inst.num_vehicles):
                    if loads[vehicle] + inst.demand[customer] > inst.capacities[vehicle]:
                        continue
                    route = plan[vehicle]
                    for pos in range(len(route) + 1):
                        trial = route[:pos] + [customer] + route[pos:]
                        if simulate_route(inst, trial, vehicle) is not None:
                            val = route_cost(inst, trial, vehicle) or 0
                            delta = val - costs[vehicle]
                            options.append((delta, vehicle, pos, val))

                if not options:
                    return False

                options.sort(key=lambda x: x[0])
                cost_best = options[0][0]
                cost_k = options[min(1, len(options) - 1)][0]
                regret = cost_k - cost_best

                if regret > max_regret or best_customer == -1:
                    max_regret = regret
                    best_customer = customer
                    best_insertion = (options[0][1], options[0][2], options[0][3])

            if best_customer == -1 or best_insertion is None:
                return False

            v, pos, new_cost = best_insertion
            plan[v].insert(pos, best_customer)
            costs[v] = new_cost
            loads[v] += inst.demand[best_customer]
            where[best_customer] = v
            removed.remove(best_customer)

        return True

    def _fast_descend(self, plan: List[List[int]], costs: List[int], where: Dict[int, int]) -> None:
        improved = True
        while improved and self._remaining() > RESERVE:
            improved = False
            if self._relocate_pass(plan, costs, where, 1):
                improved = True
            if self._relocate_pass(plan, costs, where, 2):
                improved = True
            if self._two_opt_pass(plan, costs):
                improved = True

    def _relocate_pass(self, plan: List[List[int]], costs: List[int], where: Dict[int, int], length: int) -> bool:
        inst = self.inst
        changed = False
        for source in range(len(plan)):
            position = 0
            while position + length <= len(plan[source]):
                if self._remaining() <= RESERVE:
                    return changed
                route = plan[source]
                segment = route[position : position + length]
                remainder = route[:position] + route[position + length :]
                new_source_cost = 0
                if remainder:
                    val = route_cost(inst, remainder, source)
                    if val is None:
                        position += 1
                        continue
                    new_source_cost = val

                best = None
                for neighbour in self.neighbours[segment[0]]:
                    target = where.get(neighbour)
                    if target is None:
                        continue
                    target_route = remainder if target == source else plan[target]
                    try:
                        index = target_route.index(neighbour)
                    except ValueError:
                        continue
                    for insert_at in (index, index + 1):
                        candidate = target_route[:insert_at] + segment + target_route[insert_at:]
                        val = route_cost(inst, candidate, target)
                        if val is None:
                            continue
                        delta = (
                            (val - costs[source])
                            if target == source
                            else ((new_source_cost + val) - (costs[source] + costs[target]))
                        )
                        if delta < 0 and (best is None or delta < best[0]):
                            best = (delta, target, candidate, val)

                if best is None:
                    position += 1
                    continue

                _, target, candidate, val = best
                if target == source:
                    plan[source] = candidate
                    costs[source] = val
                else:
                    plan[source] = remainder
                    costs[source] = new_source_cost
                    plan[target] = candidate
                    costs[target] = val
                    for customer in segment:
                        where[customer] = target
                changed = True
                self._offer(plan, force=False)
        return changed

    def _two_opt_pass(self, plan: List[List[int]], costs: List[int]) -> bool:
        changed = False
        for vehicle in range(len(plan)):
            route = plan[vehicle]
            if len(route) < 4:
                continue
            for i in range(len(route) - 2):
                if self._remaining() <= RESERVE:
                    return changed
                for j in range(i + 2, min(len(route), i + 12)):
                    candidate = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                    val = route_cost(self.inst, candidate, vehicle)
                    if val is not None and val < costs[vehicle]:
                        plan[vehicle] = route = candidate
                        costs[vehicle] = val
                        changed = True
                        self._offer(plan, force=False)
        return changed


Router_ = EnhancedSavingsRouter
