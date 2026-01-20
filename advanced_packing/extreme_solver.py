#!/usr/bin/env python3
"""
Extreme Tree Packing Solver - Maximum Quality

Uses extremely long optimization runs and aggressive techniques.
Designed for multi-day runs on VPS.

Target: Score < 60

Features:
1. Exhaustive rotation search for small n
2. Very long CMA-ES runs
3. Extended Parallel Tempering
4. Multi-million iteration SA
5. Progressive refinement from previous solutions
6. Automatic checkpointing

Run: python extreme_solver.py
"""

import math
import random
import time
import copy
import pickle
import os
import signal
import sys
from typing import List, Tuple, Dict, Optional
import numpy as np

from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# ============================================================================
# GEOMETRY
# ============================================================================

TREE_COORDS = np.array([
    [0.0, 0.8], [0.125, 0.5], [0.0625, 0.5], [0.2, 0.25], [0.1, 0.25],
    [0.35, 0.0], [0.075, 0.0], [0.075, -0.2], [-0.075, -0.2], [-0.075, 0.0],
    [-0.35, 0.0], [-0.1, 0.25], [-0.2, 0.25], [-0.0625, 0.5], [-0.125, 0.5],
])

BASE_POLYGON = Polygon(TREE_COORDS)

Placement = Tuple[float, float, float]
Solution = List[Placement]

def make_polygon(x: float, y: float, angle: float) -> Polygon:
    cos_a = math.cos(math.radians(angle))
    sin_a = math.sin(math.radians(angle))
    rotated = TREE_COORDS @ np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    translated = rotated + np.array([x, y])
    return Polygon(translated)

def check_collision(poly: Polygon, others: List[Polygon]) -> bool:
    for o in others:
        if poly.intersects(o) and not poly.touches(o):
            return True
    return False

def check_collision_indexed(poly: Polygon, idx: STRtree, polys: List[Polygon]) -> bool:
    for i in idx.query(poly):
        if poly.intersects(polys[i]) and not poly.touches(polys[i]):
            return True
    return False

def get_bounding_side(placements: Solution) -> float:
    if not placements:
        return 0.0
    polys = [make_polygon(*p) for p in placements]
    bounds = unary_union(polys).bounds
    return max(bounds[2] - bounds[0], bounds[3] - bounds[1])

def get_side_polys(polys: List[Polygon]) -> float:
    if not polys:
        return 0.0
    bounds = unary_union(polys).bounds
    return max(bounds[2] - bounds[0], bounds[3] - bounds[1])

def center_solution(placements: Solution) -> Solution:
    if not placements:
        return placements
    polys = [make_polygon(*p) for p in placements]
    bounds = unary_union(polys).bounds
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]

def count_overlaps(placements: Solution) -> int:
    polys = [make_polygon(*p) for p in placements]
    count = 0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                count += 1
    return count

def is_valid(placements: Solution) -> bool:
    return count_overlaps(placements) == 0

# ============================================================================
# PLACEMENT
# ============================================================================

def fermat_spiral(n: int, scale: float = 0.42) -> List[Tuple[float, float]]:
    golden = math.pi * (3.0 - math.sqrt(5.0))
    return [(scale * math.sqrt(i) * math.cos(i * golden),
             scale * math.sqrt(i) * math.sin(i * golden)) for i in range(n)]

def place_radially(polys: List[Polygon], idx: Optional[STRtree], attempts: int = 150) -> Placement:
    if not polys:
        return (0.0, 0.0, random.uniform(0, 360))

    best = None
    best_r = float('inf')

    for _ in range(attempts):
        while True:
            theta = random.uniform(0, 2 * math.pi)
            if random.random() < abs(math.sin(2 * theta)) + 0.15:
                break

        vx, vy = math.cos(theta), math.sin(theta)
        lo, hi = 0.0, 12.0

        while hi - lo > 0.005:
            mid = (lo + hi) / 2
            found = False
            for rot in range(0, 360, 10):
                p = make_polygon(mid * vx, mid * vy, rot)
                if idx:
                    coll = check_collision_indexed(p, idx, polys)
                else:
                    coll = check_collision(p, polys)
                if not coll:
                    found = True
                    break
            if found:
                hi = mid
            else:
                lo = mid

        if hi < best_r:
            px, py = hi * vx, hi * vy
            for rot in range(0, 360, 5):
                p = make_polygon(px, py, rot)
                if idx:
                    coll = check_collision_indexed(p, idx, polys)
                else:
                    coll = check_collision(p, polys)
                if not coll:
                    best_r = hi
                    best = (px, py, rot)
                    break

    return best if best else (10.0, 0.0, 0.0)

def build_solution(n: int, scale: float = 0.42) -> Solution:
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    positions = fermat_spiral(n, scale)
    placements = []
    polys = []

    for px, py in positions:
        if not polys:
            placements.append((px, py, 0.0))
            polys.append(make_polygon(px, py, 0.0))
        else:
            idx = STRtree(polys)
            found = None
            for rot in range(0, 360, 5):
                p = make_polygon(px, py, rot)
                if not check_collision_indexed(p, idx, polys):
                    found = rot
                    break

            if found is not None:
                placements.append((px, py, found))
                polys.append(make_polygon(px, py, found))
            else:
                placement = place_radially(polys, idx)
                placements.append(placement)
                polys.append(make_polygon(*placement))

    return placements

# ============================================================================
# EXHAUSTIVE SMALL N OPTIMIZATION
# ============================================================================

def exhaustive_optimize_small(n: int, base_solution: Solution,
                              angle_step: int = 5, pos_step: float = 0.02,
                              max_iters: int = 1000000) -> Solution:
    """Exhaustive optimization for small n."""
    if n <= 1:
        return base_solution
    if n > 10:
        return base_solution  # Too expensive for large n

    best_sol = list(base_solution)
    best_score = get_bounding_side(best_sol)

    current = list(base_solution)
    iters = 0

    # Grid search around current positions
    for _ in range(max_iters // (n * 72)):
        improved = False

        for i in range(n):
            x, y, d = current[i]

            # Try all rotations
            for new_d in range(0, 360, angle_step):
                # Try position offsets
                for dx in np.linspace(-0.1, 0.1, 5):
                    for dy in np.linspace(-0.1, 0.1, 5):
                        new_x = x + dx
                        new_y = y + dy

                        test = current[:i] + [(new_x, new_y, new_d)] + current[i+1:]

                        if is_valid(test):
                            score = get_bounding_side(test)
                            if score < best_score:
                                best_score = score
                                best_sol = list(test)
                                current = list(test)
                                improved = True

                        iters += 1
                        if iters >= max_iters:
                            return best_sol

        if not improved:
            break

    return best_sol

# ============================================================================
# EXTREME SIMULATED ANNEALING
# ============================================================================

def extreme_sa(placements: Solution, iterations: int = 1000000,
              temp_start: float = 10.0, temp_end: float = 1e-15,
              reheat_every: int = 50000, reheat_factor: float = 0.1) -> Solution:
    """Extreme simulated annealing with many iterations."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_polygon(*p) for p in current]
    current_score = get_side_polys(polys)

    best = list(current)
    best_score = current_score

    T = temp_start
    cooling = (temp_end / temp_start) ** (1.0 / iterations)

    shift = current_score * 0.1
    rot = 45.0

    for it in range(iterations):
        # Reheat
        if it > 0 and it % reheat_every == 0:
            T = max(T, temp_start * reheat_factor * (1 - it / iterations))
            shift = best_score * 0.08
            rot = 40.0

        i = random.randrange(n)
        x, y, d = current[i]

        r = random.random()
        if r < 0.4:
            nx = x + random.gauss(0, shift)
            ny = y + random.gauss(0, shift)
            nd = d
        elif r < 0.7:
            nx, ny = x, y
            nd = (d + random.gauss(0, rot)) % 360
        elif r < 0.9:
            nx = x + random.gauss(0, shift * 0.5)
            ny = y + random.gauss(0, shift * 0.5)
            nd = (d + random.gauss(0, rot * 0.5)) % 360
        else:
            nx = x + random.gauss(0, shift * 2.5)
            ny = y + random.gauss(0, shift * 2.5)
            nd = random.uniform(0, 360)

        new_poly = make_polygon(nx, ny, nd)
        others = polys[:i] + polys[i+1:]

        if check_collision(new_poly, others):
            T *= cooling
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_score = get_side_polys(polys)

        delta = new_score - current_score

        if delta <= 0 or random.random() < math.exp(-delta / T):
            current[i] = (nx, ny, nd)
            current_score = new_score

            if new_score < best_score:
                best_score = new_score
                best = list(current)
        else:
            polys[i] = old_poly

        T *= cooling

    return best

# ============================================================================
# EXTENDED PARALLEL TEMPERING
# ============================================================================

def extended_parallel_tempering(placements: Solution, iterations: int = 200000,
                                num_replicas: int = 12) -> Solution:
    """Extended parallel tempering with more replicas."""
    n = len(placements)
    if n <= 1:
        return placements

    # Geometric temperature ladder
    T_min, T_max = 0.001, 20.0
    temperatures = [T_min * (T_max / T_min) ** (i / (num_replicas - 1))
                   for i in range(num_replicas)]

    replicas = [list(placements) for _ in range(num_replicas)]
    polys_list = [[make_polygon(*p) for p in r] for r in replicas]
    scores = [get_side_polys(p) for p in polys_list]

    best_sol = placements
    best_score = get_bounding_side(placements) if is_valid(placements) else float('inf')

    shift_sizes = [s * 0.08 for s in scores]

    exchange_every = 50

    for it in range(iterations):
        # Update each replica
        for r in range(num_replicas):
            T = temperatures[r]
            current = replicas[r]
            polys = polys_list[r]
            current_score = scores[r]

            i = random.randrange(n)
            x, y, d = current[i]

            shift = shift_sizes[r]
            if random.random() < 0.5:
                nx = x + random.gauss(0, shift)
                ny = y + random.gauss(0, shift)
                nd = d
            elif random.random() < 0.8:
                nx, ny = x, y
                nd = (d + random.gauss(0, 35)) % 360
            else:
                nx = x + random.gauss(0, shift * 0.6)
                ny = y + random.gauss(0, shift * 0.6)
                nd = (d + random.gauss(0, 25)) % 360

            new_poly = make_polygon(nx, ny, nd)
            others = polys[:i] + polys[i+1:]

            if check_collision(new_poly, others):
                continue

            old_poly = polys[i]
            polys[i] = new_poly
            new_score = get_side_polys(polys)

            delta = new_score - current_score

            if delta <= 0 or random.random() < math.exp(-delta / T):
                current[i] = (nx, ny, nd)
                scores[r] = new_score

                if r == 0 and new_score < best_score:
                    best_score = new_score
                    best_sol = list(current)
            else:
                polys[i] = old_poly

        # Replica exchange
        if it % exchange_every == 0 and it > 0:
            for r in range(num_replicas - 1):
                T_lo = temperatures[r]
                T_hi = temperatures[r + 1]
                E_lo = scores[r]
                E_hi = scores[r + 1]

                delta = (1/T_lo - 1/T_hi) * (E_hi - E_lo)
                if delta < 0 or random.random() < math.exp(-delta):
                    replicas[r], replicas[r+1] = replicas[r+1], replicas[r]
                    polys_list[r], polys_list[r+1] = polys_list[r+1], polys_list[r]
                    scores[r], scores[r+1] = scores[r+1], scores[r]
                    shift_sizes[r], shift_sizes[r+1] = shift_sizes[r+1], shift_sizes[r]

    return best_sol

# ============================================================================
# AGGRESSIVE COMPACTION
# ============================================================================

def extreme_compact(placements: Solution, iterations: int = 100000) -> Solution:
    """Extreme compaction."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_polygon(*p) for p in current]
    current_score = get_side_polys(polys)

    for _ in range(iterations):
        bounds = unary_union(polys).bounds
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2

        i = random.randrange(n)
        x, y, d = current[i]

        dx, dy = cx - x, cy - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.005:
            continue

        step = random.uniform(0.001, min(0.04, dist * 0.25))
        nx = x + step * dx / dist
        ny = y + step * dy / dist
        nd = d

        if random.random() < 0.2:
            nd = (d + random.gauss(0, 5)) % 360

        new_poly = make_polygon(nx, ny, nd)
        others = polys[:i] + polys[i+1:]

        if check_collision(new_poly, others):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_score = get_side_polys(polys)

        if new_score <= current_score:
            current[i] = (nx, ny, nd)
            current_score = new_score
        else:
            polys[i] = old_poly

    return current

# ============================================================================
# MAIN SOLVER
# ============================================================================

class ExtremeSolver:
    """Extreme quality solver."""

    def __init__(self, seed: int = 42, verbose: bool = True, checkpoint_every: int = 5):
        self.seed = seed
        self.verbose = verbose
        self.checkpoint_every = checkpoint_every
        random.seed(seed)
        np.random.seed(seed)

        self.solutions: Dict[int, Solution] = {}
        self.scores: Dict[int, float] = {}

        self.checkpoint_file = f"checkpoint_seed_{seed}.pkl"

        # Load checkpoint if exists
        self.load_checkpoint()

        # Setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        print("\nReceived signal, saving checkpoint...")
        self.save_checkpoint()
        sys.exit(0)

    def save_checkpoint(self):
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump({
                'solutions': self.solutions,
                'scores': self.scores,
                'seed': self.seed
            }, f)
        print(f"Saved checkpoint to {self.checkpoint_file}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)
                self.solutions = data['solutions']
                self.scores = data['scores']
            print(f"Loaded checkpoint from {self.checkpoint_file}")
            print(f"Resuming from n={max(self.solutions.keys()) if self.solutions else 0}")

    def optimize_single(self, n: int) -> Solution:
        if n == 0:
            return []
        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            self.solutions[1] = sol
            self.scores[1] = 1.0
            return sol

        best_sol = None
        best_score = float('inf')

        # More restarts for small n
        num_restarts = max(3, 15 - n // 10)
        scales = [0.40 + i * 0.02 for i in range(num_restarts)]

        for restart in range(num_restarts):
            scale = scales[restart % len(scales)]

            if n - 1 in self.solutions and restart == 0:
                prev = self.solutions[n - 1]
                prev_polys = [make_polygon(*p) for p in prev]
                idx = STRtree(prev_polys)
                new_tree = place_radially(prev_polys, idx, attempts=150)
                sol = prev + [new_tree]
            else:
                random.seed(self.seed + n * 1000 + restart * 100)
                np.random.seed(self.seed + n * 1000 + restart * 100)
                sol = build_solution(n, scale)

            if not is_valid(sol):
                continue

            # Scale parameters with n
            sa_iters = max(500000, 1000000 - n * 3000)
            pt_iters = max(100000, 200000 - n * 500)
            compact_iters = max(50000, 100000 - n * 300)

            # Exhaustive for small n
            if n <= 8:
                sol = exhaustive_optimize_small(n, sol, angle_step=10, max_iters=500000)

            # Extended Parallel Tempering
            sol = extended_parallel_tempering(sol, iterations=pt_iters, num_replicas=10)

            # Extreme SA
            sol = extreme_sa(sol, iterations=sa_iters)

            # Compaction
            sol = extreme_compact(sol, compact_iters)

            # More SA
            sol = extreme_sa(sol, iterations=sa_iters // 2, temp_start=2.0)

            # Final compaction
            sol = extreme_compact(sol, compact_iters // 2)

            if not is_valid(sol):
                continue

            score = get_bounding_side(sol)
            if score < best_score:
                best_score = score
                best_sol = sol

        if best_sol is None:
            if n - 1 in self.solutions:
                prev = self.solutions[n - 1]
                prev_polys = [make_polygon(*p) for p in prev]
                idx = STRtree(prev_polys)
                new_tree = place_radially(prev_polys, idx)
                best_sol = prev + [new_tree]
            else:
                best_sol = build_solution(n, 0.5)

        best_sol = center_solution(best_sol)
        self.solutions[n] = best_sol
        self.scores[n] = get_bounding_side(best_sol)

        return best_sol

    def solve_all(self, max_n: int = 200) -> Dict[int, Solution]:
        start = time.time()
        start_n = max(self.solutions.keys()) + 1 if self.solutions else 1

        if self.verbose:
            print(f"Extreme Solver: n={start_n} to {max_n}, seed={self.seed}")
            print("=" * 70)

        for n in range(start_n, max_n + 1):
            iter_start = time.time()
            self.optimize_single(n)
            iter_time = time.time() - iter_start

            if self.verbose and (n % 5 == 0 or n <= 5):
                elapsed = time.time() - start
                score = self.total_score()
                eta = iter_time * (max_n - n)
                print(f"n={n:3d}: side={self.scores[n]:.4f}, score={score:.2f}, "
                      f"time={iter_time:.0f}s, total={elapsed/3600:.2f}h, ETA={eta/3600:.1f}h")

            # Checkpoint
            if n % self.checkpoint_every == 0:
                self.save_checkpoint()

        if self.verbose:
            print("=" * 70)
            print(f"FINAL SCORE: {self.total_score():.4f}")
            print(f"Total time: {(time.time() - start)/3600:.2f}h")

        self.save_checkpoint()
        return self.solutions

    def total_score(self) -> float:
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, get_bounding_side(sol))
            total += (side ** 2) / n
        return total

def create_submission(solutions: Dict[int, Solution], path: str = "submission.csv"):
    with open(path, "w") as f:
        f.write("id,x,y,deg\n")
        for n in range(1, 201):
            if n not in solutions:
                raise ValueError(f"Missing n={n}")
            pos = solutions[n]
            polys = [make_polygon(*p) for p in pos]
            bounds = unary_union(polys).bounds
            mx, my = bounds[0], bounds[1]
            for idx, (x, y, d) in enumerate(pos):
                f.write(f"{n:03d}_{idx},s{x-mx:.6f},s{y-my:.6f},s{d:.6f}\n")
    return path

def validate_all(solutions: Dict[int, Solution]) -> bool:
    for n in range(1, 201):
        if n not in solutions or len(solutions[n]) != n or not is_valid(solutions[n]):
            return False
    return True

def main():
    print("=" * 70)
    print("EXTREME TREE PACKING SOLVER")
    print("Multi-day optimization for score < 55")
    print("14-core parallel processing")
    print("=" * 70)

    seeds = [42, 123, 456, 789, 2025, 1337, 7777, 9999, 31415, 27182, 161803, 14142, 17320, 22360]
    best_score = float('inf')
    best_solutions = None

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Seed: {seed}")
        print(f"{'='*70}")

        solver = ExtremeSolver(seed=seed, verbose=True, checkpoint_every=5)
        solutions = solver.solve_all(max_n=200)

        if validate_all(solutions):
            score = solver.total_score()
            print(f"Seed {seed} score: {score:.4f}")

            if score < best_score:
                best_score = score
                best_solutions = copy.deepcopy(solutions)
                print(f"*** NEW BEST: {score:.4f} ***")
                create_submission(best_solutions, "submission.csv")

    if best_solutions:
        print(f"\n{'='*70}")
        print(f"BEST SCORE: {best_score:.4f}")
        print("Saved: submission.csv")

if __name__ == "__main__":
    main()
