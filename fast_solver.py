#!/usr/bin/env python3
"""
fast_solver.py - Fast Tree Packing Solver for Score < 60

Optimized for speed while maintaining quality.
Uses efficient algorithms and reduced iterations.
"""

import math
import random
import time
from typing import List, Tuple, Dict, Optional

from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# Tree polygon
TREE_COORDS = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]
BASE_POLY = Polygon(TREE_COORDS)

Placement = Tuple[float, float, float]
Solution = List[Placement]

def make_poly(x: float, y: float, deg: float) -> Polygon:
    p = BASE_POLY
    if deg != 0:
        p = affinity.rotate(p, deg, origin=(0, 0))
    if x != 0 or y != 0:
        p = affinity.translate(p, xoff=x, yoff=y)
    return p

def collides(poly: Polygon, others: List[Polygon]) -> bool:
    for o in others:
        if poly.intersects(o) and not poly.touches(o):
            return True
    return False

def collides_fast(poly: Polygon, idx: STRtree, polys: List[Polygon]) -> bool:
    for i in idx.query(poly):
        if poly.intersects(polys[i]) and not poly.touches(polys[i]):
            return True
    return False

def get_side(placements: Solution) -> float:
    if not placements:
        return 0.0
    polys = [make_poly(*p) for p in placements]
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])

def side_from_polys(polys: List[Polygon]) -> float:
    if not polys:
        return 0.0
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])

def center(placements: Solution) -> Solution:
    if not placements:
        return placements
    polys = [make_poly(*p) for p in placements]
    b = unary_union(polys).bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]

def count_overlaps(placements: Solution) -> int:
    polys = [make_poly(*p) for p in placements]
    cnt = 0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                cnt += 1
    return cnt

# Fermat spiral positions
def spiral_pos(n: int, scale: float = 0.52) -> List[Tuple[float, float]]:
    golden = math.pi * (3.0 - math.sqrt(5.0))
    return [(scale * math.sqrt(i) * math.cos(i * golden),
             scale * math.sqrt(i) * math.sin(i * golden)) for i in range(n)]

# Find best tree position
def place_radial(polys: List[Polygon], idx: Optional[STRtree], attempts: int = 50) -> Placement:
    if not polys:
        return (0.0, 0.0, random.uniform(0, 360))

    best = None
    best_r = float('inf')

    for _ in range(attempts):
        while True:
            theta = random.uniform(0, 2 * math.pi)
            if random.random() < abs(math.sin(2 * theta)) + 0.2:
                break

        vx, vy = math.cos(theta), math.sin(theta)
        lo, hi = 0.0, 15.0

        while hi - lo > 0.02:
            mid = (lo + hi) / 2
            valid = False
            for rot in [0, 60, 120, 180, 240, 300]:
                poly = make_poly(mid * vx, mid * vy, rot)
                if idx:
                    coll = collides_fast(poly, idx, polys)
                else:
                    coll = collides(poly, polys)
                if not coll:
                    valid = True
                    break
            if valid:
                hi = mid
            else:
                lo = mid

        if hi < best_r:
            px, py = hi * vx, hi * vy
            best_rot = 0.0
            for rot in range(0, 360, 15):
                poly = make_poly(px, py, rot)
                if idx:
                    coll = collides_fast(poly, idx, polys)
                else:
                    coll = collides(poly, polys)
                if not coll:
                    best_rot = float(rot)
                    break
            best_r = hi
            best = (px, py, best_rot)

    return best if best else (0.0, 0.0, 0.0)

# Build initial solution
def build_initial(n: int, scale: float = 0.52) -> Solution:
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    positions = spiral_pos(n, scale)
    placements = []
    polys = []

    for px, py in positions:
        if not polys:
            rot = random.uniform(0, 360)
            placements.append((px, py, rot))
            polys.append(make_poly(px, py, rot))
        else:
            idx = STRtree(polys)
            valid_rot = None
            for rot in range(0, 360, 20):
                poly = make_poly(px, py, rot)
                if not collides_fast(poly, idx, polys):
                    valid_rot = rot
                    break

            if valid_rot is not None:
                placements.append((px, py, valid_rot))
                polys.append(make_poly(px, py, valid_rot))
            else:
                p = place_radial(polys, idx)
                placements.append(p)
                polys.append(make_poly(*p))

    return placements

# Fast simulated annealing
def fast_sa(placements: Solution, iters: int = 15000, T0: float = 1.5, Tf: float = 1e-7) -> Solution:
    n = len(placements)
    if n <= 1:
        return placements

    curr = list(placements)
    polys = [make_poly(*p) for p in curr]
    curr_side = side_from_polys(polys)

    best = list(curr)
    best_side = curr_side

    T = T0
    cool = (Tf / T0) ** (1.0 / iters)
    shift = curr_side * 0.12
    rot_size = 45.0

    for it in range(iters):
        i = random.randrange(n)
        x, y, deg = curr[i]

        r = random.random()
        if r < 0.5:
            nx = x + random.gauss(0, shift)
            ny = y + random.gauss(0, shift)
            nd = deg
        elif r < 0.8:
            nx, ny = x, y
            nd = (deg + random.gauss(0, rot_size)) % 360
        else:
            nx = x + random.gauss(0, shift * 0.7)
            ny = y + random.gauss(0, shift * 0.7)
            nd = (deg + random.gauss(0, rot_size * 0.7)) % 360

        new_poly = make_poly(nx, ny, nd)
        others = polys[:i] + polys[i+1:]

        if collides(new_poly, others):
            T *= cool
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_side = side_from_polys(polys)
        delta = new_side - curr_side

        if delta <= 0 or random.random() < math.exp(-delta / T):
            curr[i] = (nx, ny, nd)
            curr_side = new_side
            if new_side < best_side:
                best_side = new_side
                best = list(curr)
        else:
            polys[i] = old_poly

        T *= cool
        if it % 3000 == 0:
            shift = max(0.005, shift * 0.9)
            rot_size = max(2.0, rot_size * 0.9)

    return best

# Compact toward center
def compact(placements: Solution, iters: int = 2000) -> Solution:
    n = len(placements)
    if n <= 1:
        return placements

    curr = list(placements)
    polys = [make_poly(*p) for p in curr]
    curr_side = side_from_polys(polys)

    for _ in range(iters):
        b = unary_union(polys).bounds
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

        i = random.randrange(n)
        x, y, deg = curr[i]

        dx, dy = cx - x, cy - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.01:
            continue

        step = random.uniform(0.005, min(0.06, dist * 0.4))
        nx = x + step * dx / dist
        ny = y + step * dy / dist

        new_poly = make_poly(nx, ny, deg)
        others = polys[:i] + polys[i+1:]

        if collides(new_poly, others):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_side = side_from_polys(polys)

        if new_side <= curr_side:
            curr[i] = (nx, ny, deg)
            curr_side = new_side
        else:
            polys[i] = old_poly

    return curr

# Local refinement
def refine(placements: Solution, iters: int = 3000) -> Solution:
    n = len(placements)
    if n <= 1:
        return placements

    curr = list(placements)
    polys = [make_poly(*p) for p in curr]
    curr_side = side_from_polys(polys)

    for _ in range(iters):
        i = random.randrange(n)
        x, y, deg = curr[i]

        nx = x + random.gauss(0, 0.01)
        ny = y + random.gauss(0, 0.01)
        nd = (deg + random.gauss(0, 3)) % 360

        new_poly = make_poly(nx, ny, nd)
        others = polys[:i] + polys[i+1:]

        if collides(new_poly, others):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_side = side_from_polys(polys)

        if new_side < curr_side:
            curr[i] = (nx, ny, nd)
            curr_side = new_side
        else:
            polys[i] = old_poly

    return curr

class FastSolver:
    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        random.seed(seed)
        self.solutions: Dict[int, Solution] = {}
        self.scores: Dict[int, float] = {}

    def solve_n(self, n: int) -> Solution:
        if n == 0:
            return []
        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            self.solutions[1] = sol
            self.scores[1] = get_side(sol)
            return sol

        best_sol = None
        best_side = float('inf')

        for attempt in range(2):
            random.seed(self.seed + n * 100 + attempt * 17)

            if n - 1 in self.solutions and attempt == 0:
                prev = self.solutions[n - 1]
                prev_polys = [make_poly(*p) for p in prev]
                idx = STRtree(prev_polys)
                new_tree = place_radial(prev_polys, idx, attempts=60)
                sol = prev + [new_tree]
            else:
                scale = 0.50 + attempt * 0.04
                sol = build_initial(n, scale)

            if count_overlaps(sol) > 0:
                continue

            # Optimize - scale with n
            sa_iters = min(20000, 8000 + n * 60)
            comp_iters = min(3000, 1000 + n * 10)

            sol = fast_sa(sol, iters=sa_iters)
            sol = compact(sol, iters=comp_iters)
            sol = refine(sol, iters=comp_iters)
            sol = fast_sa(sol, iters=sa_iters // 2, T0=0.5)
            sol = compact(sol, iters=comp_iters // 2)

            if count_overlaps(sol) > 0:
                continue

            side = get_side(sol)
            if side < best_side:
                best_side = side
                best_sol = sol

        if best_sol is None:
            if n - 1 in self.solutions:
                prev = self.solutions[n - 1]
                prev_polys = [make_poly(*p) for p in prev]
                idx = STRtree(prev_polys)
                new_tree = place_radial(prev_polys, idx)
                best_sol = prev + [new_tree]
            else:
                best_sol = build_initial(n)

        best_sol = center(best_sol)
        self.solutions[n] = best_sol
        self.scores[n] = get_side(best_sol)
        return best_sol

    def solve_all(self, max_n: int = 200) -> Dict[int, Solution]:
        start = time.time()

        if self.verbose:
            print(f"Fast Solver: n=1 to {max_n}")
            print("=" * 60)

        for n in range(1, max_n + 1):
            self.solve_n(n)

            if self.verbose and (n % 20 == 0 or n <= 5):
                elapsed = time.time() - start
                score = self.total_score()
                print(f"n={n:3d}: side={self.scores[n]:.4f}, score={score:.2f}, time={elapsed:.1f}s")

        if self.verbose:
            print("=" * 60)
            print(f"Final Score: {self.total_score():.4f}")
            print(f"Time: {time.time() - start:.1f}s")

        return self.solutions

    def total_score(self) -> float:
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, get_side(sol))
            total += (side ** 2) / n
        return total

def create_submission(solutions: Dict[int, Solution], path: str = "submission.csv") -> str:
    with open(path, "w") as f:
        f.write("id,x,y,deg\n")
        for n in range(1, 201):
            if n not in solutions:
                raise ValueError(f"Missing n={n}")
            positions = solutions[n]
            polys = [make_poly(*p) for p in positions]
            b = unary_union(polys).bounds
            min_x, min_y = b[0], b[1]
            for idx, (x, y, deg) in enumerate(positions):
                f.write(f"{n:03d}_{idx},s{x - min_x:.6f},s{y - min_y:.6f},s{deg:.6f}\n")
    return path

def validate(solutions: Dict[int, Solution]) -> Tuple[bool, List[str]]:
    errors = []
    for n in range(1, 201):
        if n not in solutions:
            errors.append(f"Missing n={n}")
        elif len(solutions[n]) != n:
            errors.append(f"n={n}: wrong count")
        elif count_overlaps(solutions[n]) > 0:
            errors.append(f"n={n}: overlaps")
    return len(errors) == 0, errors

def main():
    print("=" * 60)
    print("FAST TREE PACKING SOLVER")
    print("=" * 60)

    best_score = float('inf')
    best_solutions = None

    seeds = [42, 123, 456]

    for seed in seeds:
        print(f"\nSeed: {seed}")
        print("-" * 40)

        solver = FastSolver(seed=seed, verbose=True)
        solutions = solver.solve_all(max_n=200)

        valid, errors = validate(solutions)
        if not valid:
            print(f"Errors: {errors[:3]}")
            continue

        score = solver.total_score()
        if score < best_score:
            best_score = score
            best_solutions = solutions
            print(f"*** Best: {score:.4f} ***")

    if best_solutions:
        print(f"\n{'='*60}")
        print(f"BEST SCORE: {best_score:.4f}")
        path = create_submission(best_solutions, "submission.csv")
        print(f"Saved: {path}")

        valid, _ = validate(best_solutions)
        print(f"Valid: {valid}")

if __name__ == "__main__":
    main()
