"""
ultra_solver.py - Ultra-optimized Tree Packing Solver

Implements aggressive optimization to achieve score < 60:
1. Intelligent incremental building with optimal placement
2. Very aggressive simulated annealing with reheat
3. Global optimization with basin hopping
4. Specialized small-n optimization
5. Multi-stage refinement pipeline
"""

import math
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from copy import deepcopy
import time
import multiprocessing as mp
from functools import partial

from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# ============================================================================
# GEOMETRY
# ============================================================================

TREE_COORDS = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]
BASE_POLYGON = Polygon(TREE_COORDS)
TREE_WIDTH = 0.7
TREE_HEIGHT = 1.0
TREE_AREA = BASE_POLYGON.area  # ~0.45

Placement = Tuple[float, float, float]
Solution = List[Placement]

def make_poly(x: float, y: float, deg: float) -> Polygon:
    """Create tree polygon."""
    p = BASE_POLYGON
    if deg != 0:
        p = affinity.rotate(p, deg, origin=(0, 0))
    if x != 0 or y != 0:
        p = affinity.translate(p, xoff=x, yoff=y)
    return p

def has_collision(poly: Polygon, others: List[Polygon]) -> bool:
    """Check collision (touching OK)."""
    for o in others:
        if poly.intersects(o) and not poly.touches(o):
            return True
    return False

def fast_collision(poly: Polygon, idx: STRtree, polys: List[Polygon]) -> bool:
    """Fast collision check."""
    for i in idx.query(poly):
        if poly.intersects(polys[i]) and not poly.touches(polys[i]):
            return True
    return False

def get_side(placements: Solution) -> float:
    """Get bounding square side."""
    if not placements:
        return 0.0
    polys = [make_poly(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])

def get_side_polys(polys: List[Polygon]) -> float:
    """Get side from polygons."""
    if not polys:
        return 0.0
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])

def center(placements: Solution) -> Solution:
    """Center around origin."""
    if not placements:
        return placements
    polys = [make_poly(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]

def check_overlaps(placements: Solution) -> int:
    """Count overlapping pairs."""
    polys = [make_poly(x, y, d) for x, y, d in placements]
    count = 0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                count += 1
    return count

# ============================================================================
# OPTIMAL INITIAL PLACEMENT
# ============================================================================

def optimal_single_tree() -> Solution:
    """Best placement for n=1."""
    return [(0.0, 0.0, 0.0)]

def optimal_two_trees() -> Solution:
    """Optimized placement for n=2."""
    best_sol = None
    best_side = float('inf')

    # Try different configurations
    for angle1 in range(0, 360, 15):
        for angle2 in range(0, 360, 15):
            for sep in np.linspace(0.5, 1.5, 20):
                for theta in range(0, 360, 30):
                    dx = sep * math.cos(math.radians(theta))
                    dy = sep * math.sin(math.radians(theta))

                    sol = [(0.0, 0.0, angle1), (dx, dy, angle2)]

                    if check_overlaps(sol) > 0:
                        continue

                    side = get_side(sol)
                    if side < best_side:
                        best_side = side
                        best_sol = sol

    return center(best_sol) if best_sol else [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]

def fermat_spiral(n: int, scale: float = 0.5) -> List[Tuple[float, float]]:
    """Fermat spiral positions."""
    golden = math.pi * (3 - math.sqrt(5))
    positions = []
    for i in range(n):
        r = scale * math.sqrt(i)
        theta = i * golden
        positions.append((r * math.cos(theta), r * math.sin(theta)))
    return positions

def hexagonal_grid(n: int, spacing: float = 0.8) -> List[Tuple[float, float]]:
    """Hexagonal grid positions."""
    positions = [(0.0, 0.0)]
    if n <= 1:
        return positions[:n]

    dx = spacing * 0.7
    dy = spacing * 0.7 * 0.866

    ring = 1
    while len(positions) < n:
        for i in range(6 * ring):
            if len(positions) >= n:
                break
            side = i // ring
            step = i % ring
            angle = math.pi / 3 * side

            # Position on this ring
            x = ring * dx * math.cos(angle) - step * dx * math.cos(angle + math.pi / 3)
            y = ring * dy * math.sin(angle) - step * dy * math.sin(angle + math.pi / 3)
            positions.append((x, y))
        ring += 1

    return positions[:n]

# ============================================================================
# ULTRA SIMULATED ANNEALING
# ============================================================================

def ultra_sa(placements: Solution,
             iterations: int = 100000,
             temp_init: float = 3.0,
             temp_final: float = 1e-8,
             reheat_interval: int = 20000,
             reheat_factor: float = 0.3) -> Solution:
    """Ultra-aggressive SA with periodic reheating."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_poly(x, y, d) for x, y, d in current]
    current_side = get_side_polys(polys)

    best = list(current)
    best_side = current_side

    T = temp_init
    base_cooling = (temp_final / temp_init) ** (1.0 / iterations)

    max_shift = current_side * 0.2
    max_rot = 60.0

    stuck_count = 0
    last_improvement = 0

    for it in range(iterations):
        # Periodic reheat
        if it > 0 and it % reheat_interval == 0:
            T = max(T, temp_init * reheat_factor)
            max_shift = best_side * 0.15
            max_rot = 45.0

        # Random tree
        i = random.randrange(n)
        x, y, deg = current[i]

        # Move type selection with adaptive probabilities
        progress = it / iterations
        r = random.random()

        if r < 0.45:
            # Translation
            new_x = x + random.gauss(0, max_shift)
            new_y = y + random.gauss(0, max_shift)
            new_deg = deg
        elif r < 0.75:
            # Rotation
            new_x, new_y = x, y
            new_deg = (deg + random.gauss(0, max_rot)) % 360
        elif r < 0.9:
            # Combined
            new_x = x + random.gauss(0, max_shift * 0.7)
            new_y = y + random.gauss(0, max_shift * 0.7)
            new_deg = (deg + random.gauss(0, max_rot * 0.7)) % 360
        else:
            # Large jump
            new_x = x + random.gauss(0, max_shift * 2)
            new_y = y + random.gauss(0, max_shift * 2)
            new_deg = random.uniform(0, 360)

        # Validate
        new_poly = make_poly(new_x, new_y, new_deg)
        others = polys[:i] + polys[i+1:]

        if has_collision(new_poly, others):
            T *= base_cooling
            stuck_count += 1
            continue

        # Score new configuration
        old_poly = polys[i]
        polys[i] = new_poly
        new_side = get_side_polys(polys)

        delta = new_side - current_side

        if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            current[i] = (new_x, new_y, new_deg)
            current_side = new_side
            stuck_count = 0

            if new_side < best_side:
                best_side = new_side
                best = list(current)
                last_improvement = it
        else:
            polys[i] = old_poly

        T *= base_cooling

        # Adaptive step size
        if stuck_count > 100:
            max_shift *= 0.95
            max_rot *= 0.95
            stuck_count = 0
        elif it - last_improvement > 5000:
            max_shift = max(0.005, max_shift * 0.99)
            max_rot = max(1.0, max_rot * 0.99)

    return best

# ============================================================================
# COMPACTION
# ============================================================================

def aggressive_compact(placements: Solution, iterations: int = 10000) -> Solution:
    """Aggressive compaction toward center."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_poly(x, y, d) for x, y, d in current]
    current_side = get_side_polys(polys)

    for _ in range(iterations):
        b = unary_union(polys).bounds
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

        i = random.randrange(n)
        x, y, deg = current[i]

        # Move toward center with random step
        dx, dy = cx - x, cy - y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < 0.01:
            continue

        step = random.uniform(0.005, min(0.1, dist))
        new_x = x + step * dx / dist
        new_y = y + step * dy / dist

        # Also try slight rotation
        new_deg = deg
        if random.random() < 0.3:
            new_deg = (deg + random.gauss(0, 10)) % 360

        new_poly = make_poly(new_x, new_y, new_deg)
        others = polys[:i] + polys[i+1:]

        if has_collision(new_poly, others):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_side = get_side_polys(polys)

        if new_side <= current_side:
            current[i] = (new_x, new_y, new_deg)
            current_side = new_side
        else:
            polys[i] = old_poly

    return current

def squeeze_bounds(placements: Solution, iterations: int = 5000) -> Solution:
    """Squeeze boundary trees inward."""
    n = len(placements)
    if n <= 2:
        return placements

    current = list(placements)
    polys = [make_poly(x, y, d) for x, y, d in current]

    for _ in range(iterations):
        b = unary_union(polys).bounds
        minx, miny, maxx, maxy = b
        current_side = max(maxx - minx, maxy - miny)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

        # Find boundary trees
        boundary = []
        for i, (x, y, d) in enumerate(current):
            pb = polys[i].bounds
            on_edge = (abs(pb[0] - minx) < 0.02 or abs(pb[2] - maxx) < 0.02 or
                      abs(pb[1] - miny) < 0.02 or abs(pb[3] - maxy) < 0.02)
            if on_edge:
                boundary.append(i)

        if not boundary:
            break

        i = random.choice(boundary)
        x, y, deg = current[i]

        # Move toward center
        dx, dy = cx - x, cy - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.01:
            continue

        step = random.uniform(0.005, 0.03)
        new_x = x + step * dx / dist
        new_y = y + step * dy / dist

        new_poly = make_poly(new_x, new_y, deg)
        others = polys[:i] + polys[i+1:]

        if has_collision(new_poly, others):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_side = get_side_polys(polys)

        if new_side < current_side:
            current[i] = (new_x, new_y, deg)
        else:
            polys[i] = old_poly

    return current

# ============================================================================
# ROTATION OPTIMIZATION
# ============================================================================

def optimize_rotations(placements: Solution, iterations: int = 5000) -> Solution:
    """Optimize rotations for better packing."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_poly(x, y, d) for x, y, d in current]
    current_side = get_side_polys(polys)

    for _ in range(iterations):
        i = random.randrange(n)
        x, y, deg = current[i]

        # Try new rotation
        new_deg = (deg + random.gauss(0, 30)) % 360

        new_poly = make_poly(x, y, new_deg)
        others = polys[:i] + polys[i+1:]

        if has_collision(new_poly, others):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_side = get_side_polys(polys)

        if new_side < current_side:
            current[i] = (x, y, new_deg)
            current_side = new_side
        else:
            polys[i] = old_poly

    return current

# ============================================================================
# LOCAL SEARCH
# ============================================================================

def local_search(placements: Solution, iterations: int = 20000) -> Solution:
    """Fine-grained local search."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_poly(x, y, d) for x, y, d in current]
    current_side = get_side_polys(polys)

    step = 0.01

    for it in range(iterations):
        i = random.randrange(n)
        x, y, deg = current[i]

        # Small adjustments
        dx = random.gauss(0, step)
        dy = random.gauss(0, step)
        drot = random.gauss(0, 3)

        new_x, new_y = x + dx, y + dy
        new_deg = (deg + drot) % 360

        new_poly = make_poly(new_x, new_y, new_deg)
        others = polys[:i] + polys[i+1:]

        if has_collision(new_poly, others):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_side = get_side_polys(polys)

        if new_side < current_side:
            current[i] = (new_x, new_y, new_deg)
            current_side = new_side
        else:
            polys[i] = old_poly

        # Adaptive step
        if it % 2000 == 0:
            step *= 0.9

    return current

# ============================================================================
# INCREMENTAL BUILDER
# ============================================================================

def find_best_position(existing_polys: List[Polygon],
                       tree_idx: Optional[STRtree],
                       n_attempts: int = 100) -> Placement:
    """Find optimal position for new tree."""
    if not existing_polys:
        return (0.0, 0.0, random.uniform(0, 360))

    best = None
    best_score = float('inf')

    for _ in range(n_attempts):
        # Random angle toward center with bias to corners
        while True:
            angle = random.uniform(0, 2 * math.pi)
            if random.random() < abs(math.sin(2 * angle)) + 0.1:
                break

        vx, vy = math.cos(angle), math.sin(angle)

        # Binary search for radius
        low, high = 0.0, 25.0
        best_r = high

        while high - low > 0.01:
            mid = (low + high) / 2
            px, py = mid * vx, mid * vy

            valid = False
            for rot in range(0, 360, 30):
                poly = make_poly(px, py, rot)
                if tree_idx:
                    coll = fast_collision(poly, tree_idx, existing_polys)
                else:
                    coll = has_collision(poly, existing_polys)

                if not coll:
                    valid = True
                    best_r = mid
                    break

            if valid:
                high = mid
            else:
                low = mid

        # Score this position
        px, py = best_r * vx, best_r * vy

        # Find best rotation
        best_rot = 0
        for rot in range(0, 360, 10):
            poly = make_poly(px, py, rot)
            if tree_idx:
                coll = fast_collision(poly, tree_idx, existing_polys)
            else:
                coll = has_collision(poly, existing_polys)

            if not coll:
                best_rot = rot
                break

        # Score: prefer smaller bounding box
        test_polys = existing_polys + [make_poly(px, py, best_rot)]
        score = get_side_polys(test_polys)

        if score < best_score:
            best_score = score
            best = (px, py, best_rot)

    return best if best else (0.0, 0.0, 0.0)

# ============================================================================
# MAIN SOLVER
# ============================================================================

class UltraSolver:
    """Ultra-optimized solver."""

    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        random.seed(seed)
        np.random.seed(seed)
        self.solutions: Dict[int, Solution] = {}
        self.scores: Dict[int, float] = {}

    def solve_n(self, n: int, attempts: int = 3) -> Solution:
        """Solve for specific n."""
        if n == 0:
            return []
        if n == 1:
            sol = optimal_single_tree()
            self.solutions[1] = sol
            self.scores[1] = get_side(sol)
            return sol
        if n == 2:
            sol = optimal_two_trees()
            self.solutions[2] = sol
            self.scores[2] = get_side(sol)
            return sol

        best_sol = None
        best_side = float('inf')

        for attempt in range(attempts):
            random.seed(self.seed + n * 1000 + attempt * 100)

            # Start from previous solution or fresh
            if n - 1 in self.solutions and attempt == 0:
                prev = self.solutions[n - 1]
                prev_polys = [make_poly(x, y, d) for x, y, d in prev]
                tree_idx = STRtree(prev_polys)
                new_tree = find_best_position(prev_polys, tree_idx, n_attempts=100)
                sol = prev + [new_tree]
            else:
                # Fresh spiral-based placement
                positions = fermat_spiral(n, scale=0.55 + attempt * 0.05)
                sol = []
                placed_polys = []

                for px, py in positions:
                    if not placed_polys:
                        rot = random.uniform(0, 360)
                        sol.append((px, py, rot))
                        placed_polys.append(make_poly(px, py, rot))
                    else:
                        tree_idx = STRtree(placed_polys)

                        # Find valid rotation at position
                        found = False
                        for rot in range(0, 360, 15):
                            poly = make_poly(px, py, rot)
                            if not fast_collision(poly, tree_idx, placed_polys):
                                sol.append((px, py, rot))
                                placed_polys.append(poly)
                                found = True
                                break

                        if not found:
                            # Use radial placement
                            new_tree = find_best_position(placed_polys, tree_idx)
                            sol.append(new_tree)
                            placed_polys.append(make_poly(*new_tree))

            # Validate
            if check_overlaps(sol) > 0:
                continue

            # Optimization pipeline
            # Scale iterations with n
            sa_iter = max(50000, 30000 + n * 300)
            compact_iter = max(5000, 3000 + n * 30)

            sol = ultra_sa(sol, iterations=sa_iter)
            sol = aggressive_compact(sol, iterations=compact_iter)
            sol = squeeze_bounds(sol, iterations=compact_iter // 2)
            sol = optimize_rotations(sol, iterations=compact_iter)
            sol = local_search(sol, iterations=compact_iter * 2)

            # Second SA pass
            sol = ultra_sa(sol, iterations=sa_iter // 2,
                          temp_init=1.0, temp_final=1e-9)

            sol = aggressive_compact(sol, iterations=compact_iter // 2)

            # Validate again
            if check_overlaps(sol) > 0:
                continue

            side = get_side(sol)
            if side < best_side:
                best_side = side
                best_sol = sol

        if best_sol is None:
            # Fallback
            if n - 1 in self.solutions:
                prev = self.solutions[n - 1]
                prev_polys = [make_poly(x, y, d) for x, y, d in prev]
                tree_idx = STRtree(prev_polys)
                new_tree = find_best_position(prev_polys, tree_idx)
                best_sol = prev + [new_tree]
            else:
                positions = fermat_spiral(n)
                best_sol = [(x, y, random.uniform(0, 360)) for x, y in positions]

        best_sol = center(best_sol)
        self.solutions[n] = best_sol
        self.scores[n] = get_side(best_sol)

        return best_sol

    def solve_all(self, max_n: int = 200) -> Dict[int, Solution]:
        """Solve all n from 1 to max_n."""
        start = time.time()

        if self.verbose:
            print(f"Ultra Solver: n=1 to {max_n}")
            print("=" * 70)

        for n in range(1, max_n + 1):
            self.solve_n(n)

            if self.verbose and (n % 10 == 0 or n <= 5):
                elapsed = time.time() - start
                score = self.total_score()
                print(f"n={n:3d}: side={self.scores[n]:.4f}, "
                      f"score={score:.2f}, time={elapsed:.1f}s")

        if self.verbose:
            print("=" * 70)
            print(f"Final Score: {self.total_score():.4f}")
            print(f"Time: {time.time() - start:.1f}s")

        return self.solutions

    def total_score(self) -> float:
        """Compute competition score."""
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, get_side(sol))
            total += (side ** 2) / n
        return total

# ============================================================================
# SUBMISSION
# ============================================================================

def create_submission(solutions: Dict[int, Solution],
                     path: str = "submission.csv") -> str:
    """Create submission file."""
    with open(path, "w") as f:
        f.write("id,x,y,deg\n")

        for n in range(1, 201):
            if n not in solutions:
                raise ValueError(f"Missing n={n}")

            positions = solutions[n]
            polys = [make_poly(x, y, d) for x, y, d in positions]
            b = unary_union(polys).bounds
            min_x, min_y = b[0], b[1]

            for idx, (x, y, deg) in enumerate(positions):
                X, Y = x - min_x, y - min_y
                f.write(f"{n:03d}_{idx},s{X:.6f},s{Y:.6f},s{deg:.6f}\n")

    return path

def validate(solutions: Dict[int, Solution]) -> Tuple[bool, List[str]]:
    """Validate solutions."""
    errors = []
    for n in range(1, 201):
        if n not in solutions:
            errors.append(f"Missing n={n}")
        elif len(solutions[n]) != n:
            errors.append(f"n={n}: wrong count")
        elif check_overlaps(solutions[n]) > 0:
            errors.append(f"n={n}: overlaps")
    return len(errors) == 0, errors

# ============================================================================
# MAIN
# ============================================================================

def run_solver(seed: int, verbose: bool = False) -> Tuple[float, Dict[int, Solution]]:
    """Run solver with given seed."""
    solver = UltraSolver(seed=seed, verbose=verbose)
    solutions = solver.solve_all(max_n=200)
    valid, _ = validate(solutions)
    if not valid:
        return float('inf'), {}
    return solver.total_score(), solutions

def main():
    print("=" * 70)
    print("ULTRA TREE PACKING SOLVER")
    print("=" * 70)

    best_score = float('inf')
    best_solutions = None

    # Try multiple seeds
    seeds = [42, 123, 456, 789, 2025, 1337, 9999, 5555]

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Seed: {seed}")
        print(f"{'='*70}")

        score, solutions = run_solver(seed, verbose=True)

        if score < best_score:
            best_score = score
            best_solutions = solutions
            print(f"*** New best: {score:.4f} ***")

    if best_solutions:
        print(f"\n{'='*70}")
        print(f"BEST SCORE: {best_score:.4f}")
        print(f"{'='*70}")

        path = create_submission(best_solutions, "submission.csv")
        print(f"Saved: {path}")

        valid, errors = validate(best_solutions)
        if valid:
            print("✓ All solutions valid!")
        else:
            print(f"✗ Errors: {errors[:5]}")
    else:
        print("ERROR: No valid solutions!")

if __name__ == "__main__":
    main()
