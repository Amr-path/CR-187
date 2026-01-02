"""
optimized_solver.py - Advanced Tree Packing Solver for Santa 2025

Implements multiple advanced optimization techniques to achieve score < 60:
1. Hexagonal grid-based initial placement
2. Adaptive simulated annealing with geometric cooling
3. Tree rotation optimization for nesting
4. Iterative compaction
5. Multi-restart with different strategies
6. Local search refinement
"""

import math
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from copy import deepcopy
import time

from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# ============================================================================
# GEOMETRY DEFINITIONS (from original geometry.py)
# ============================================================================

TREE_COORDS = [
    (0.0, 0.8),        # tip
    (0.125, 0.5),      # top tier outer right
    (0.0625, 0.5),     # top tier inner right
    (0.2, 0.25),       # middle tier outer right
    (0.1, 0.25),       # middle tier inner right
    (0.35, 0.0),       # bottom tier outer right
    (0.075, 0.0),      # trunk top right
    (0.075, -0.2),     # trunk bottom right
    (-0.075, -0.2),    # trunk bottom left
    (-0.075, 0.0),     # trunk top left
    (-0.35, 0.0),      # bottom tier outer left
    (-0.1, 0.25),      # middle tier inner left
    (-0.2, 0.25),      # middle tier outer left
    (-0.0625, 0.5),    # top tier inner left
    (-0.125, 0.5),     # top tier outer left
]

BASE_POLYGON = Polygon(TREE_COORDS)
TREE_WIDTH = 0.7
TREE_HEIGHT = 1.0

# Pre-compute tree centroid offset
TREE_CENTROID_Y = 0.3  # Approximate visual center

Placement = Tuple[float, float, float]  # (x, y, angle_deg)
Solution = List[Placement]

def make_tree_polygon(x: float, y: float, angle_deg: float) -> Polygon:
    """Create tree polygon at position with rotation."""
    poly = BASE_POLYGON
    if angle_deg != 0:
        poly = affinity.rotate(poly, angle_deg, origin=(0, 0))
    if x != 0 or y != 0:
        poly = affinity.translate(poly, xoff=x, yoff=y)
    return poly

def has_collision(tree_poly: Polygon, other_polys: List[Polygon]) -> bool:
    """Check if tree_poly overlaps any other (touching OK)."""
    for poly in other_polys:
        if tree_poly.intersects(poly) and not tree_poly.touches(poly):
            return True
    return False

def has_collision_fast(tree_poly: Polygon, tree_index: STRtree, all_polys: List[Polygon]) -> bool:
    """Fast collision check using spatial index."""
    candidates = tree_index.query(tree_poly)
    for idx in candidates:
        if tree_poly.intersects(all_polys[idx]) and not tree_poly.touches(all_polys[idx]):
            return True
    return False

def get_bounding_square(placements: Solution) -> float:
    """Compute bounding square side from placements."""
    if not placements:
        return 0.0
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    bounds = unary_union(polys).bounds
    return max(bounds[2] - bounds[0], bounds[3] - bounds[1])

def get_bounds(placements: Solution) -> Tuple[float, float, float, float]:
    """Get minx, miny, maxx, maxy."""
    if not placements:
        return (0, 0, 0, 0)
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    return unary_union(polys).bounds

def center_solution(placements: Solution) -> Solution:
    """Center placements around origin."""
    if not placements:
        return placements
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    bounds = unary_union(polys).bounds
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]

def check_all_overlaps(placements: Solution) -> List[Tuple[int, int]]:
    """Find all overlapping pairs."""
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    overlaps = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                overlaps.append((i, j))
    return overlaps

# ============================================================================
# ADVANCED PLACEMENT STRATEGIES
# ============================================================================

def hexagonal_positions(n: int, spacing: float = 0.85) -> List[Tuple[float, float]]:
    """Generate hexagonal grid positions for n trees."""
    positions = []

    # Hexagonal grid parameters
    dx = spacing * TREE_WIDTH
    dy = spacing * TREE_HEIGHT * 0.866  # sqrt(3)/2

    layer = 0
    placed = 0

    while placed < n:
        if layer == 0:
            positions.append((0.0, 0.0))
            placed += 1
        else:
            # Generate hexagonal ring
            for side in range(6):
                for step in range(layer):
                    if placed >= n:
                        break
                    angle = math.pi / 3 * side + math.pi / 6

                    # Start position for this side
                    start_x = layer * dx * math.cos(angle - math.pi / 6)
                    start_y = layer * dy * math.sin(angle - math.pi / 6)

                    # Move along the side
                    side_angle = angle + math.pi / 2
                    x = start_x + step * dx * math.cos(side_angle)
                    y = start_y + step * dy * math.sin(side_angle)

                    positions.append((x, y))
                    placed += 1
        layer += 1

    return positions[:n]

def spiral_positions(n: int, spacing: float = 0.9) -> List[Tuple[float, float]]:
    """Generate Fermat spiral positions for n trees."""
    positions = []
    golden_angle = math.pi * (3 - math.sqrt(5))  # ~137.5 degrees

    c = spacing * max(TREE_WIDTH, TREE_HEIGHT) * 0.5

    for i in range(n):
        r = c * math.sqrt(i)
        theta = i * golden_angle
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        positions.append((x, y))

    return positions

def concentric_positions(n: int, spacing: float = 0.85) -> List[Tuple[float, float]]:
    """Generate concentric circle positions."""
    positions = [(0.0, 0.0)]
    if n <= 1:
        return positions[:n]

    placed = 1
    ring = 1
    base_radius = spacing * max(TREE_WIDTH, TREE_HEIGHT)

    while placed < n:
        radius = ring * base_radius
        circumference = 2 * math.pi * radius
        trees_in_ring = max(1, int(circumference / (spacing * TREE_WIDTH)))

        for i in range(trees_in_ring):
            if placed >= n:
                break
            angle = 2 * math.pi * i / trees_in_ring
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions.append((x, y))
            placed += 1

        ring += 1

    return positions[:n]

def grid_positions(n: int, spacing: float = 0.9) -> List[Tuple[float, float]]:
    """Generate square grid positions."""
    positions = []
    side = int(math.ceil(math.sqrt(n)))

    dx = spacing * TREE_WIDTH
    dy = spacing * TREE_HEIGHT

    for i in range(side):
        for j in range(side):
            if len(positions) >= n:
                break
            x = (i - side / 2) * dx
            y = (j - side / 2) * dy
            positions.append((x, y))

    return positions[:n]

# ============================================================================
# OPTIMAL ROTATION FINDING
# ============================================================================

def find_best_rotation(x: float, y: float, other_polys: List[Polygon],
                       tree_index: Optional[STRtree] = None,
                       n_angles: int = 72) -> Optional[float]:
    """Find best rotation angle for a tree at given position."""
    best_angle = None

    for i in range(n_angles):
        angle = 360.0 * i / n_angles
        poly = make_tree_polygon(x, y, angle)

        if tree_index:
            has_coll = has_collision_fast(poly, tree_index, other_polys)
        else:
            has_coll = has_collision(poly, other_polys)

        if not has_coll:
            best_angle = angle
            break

    return best_angle

def find_optimal_rotation_pair(angle1: float, distance: float = 0.8) -> float:
    """Find optimal rotation for second tree to nest with first."""
    # Trees nest well when rotated ~180 degrees offset
    return (angle1 + 180 + random.uniform(-30, 30)) % 360

# ============================================================================
# GREEDY PLACEMENT WITH MULTIPLE STRATEGIES
# ============================================================================

def place_tree_radial(existing_polys: List[Polygon],
                      tree_index: Optional[STRtree],
                      n_attempts: int = 60,
                      start_radius: float = 20.0) -> Tuple[float, float, float]:
    """Place tree using radial approach with optimized angles."""
    if not existing_polys:
        return (0.0, 0.0, random.uniform(0, 360))

    best_placement = None
    best_radius = float('inf')

    for _ in range(n_attempts):
        # Weighted angle favoring diagonals
        while True:
            angle_rad = random.uniform(0, 2 * math.pi)
            if random.random() < abs(math.sin(2 * angle_rad)):
                break

        vx = math.cos(angle_rad)
        vy = math.sin(angle_rad)

        # Binary search for optimal radius
        low, high = 0.0, start_radius
        best_r = high

        while high - low > 0.02:
            mid = (low + high) / 2
            px, py = mid * vx, mid * vy

            # Try multiple rotations
            found_valid = False
            for rot in [0, 60, 120, 180, 240, 300]:
                poly = make_tree_polygon(px, py, rot)

                if tree_index:
                    has_coll = has_collision_fast(poly, tree_index, existing_polys)
                else:
                    has_coll = has_collision(poly, existing_polys)

                if not has_coll:
                    found_valid = True
                    best_r = mid
                    break

            if found_valid:
                high = mid
            else:
                low = mid

        if best_r < best_radius:
            best_radius = best_r
            px, py = best_r * vx, best_r * vy

            # Find best rotation at this position
            best_rot = 0.0
            for rot in range(0, 360, 5):
                poly = make_tree_polygon(px, py, rot)
                if tree_index:
                    has_coll = has_collision_fast(poly, tree_index, existing_polys)
                else:
                    has_coll = has_collision(poly, existing_polys)
                if not has_coll:
                    best_rot = rot
                    break

            best_placement = (px, py, best_rot)

    return best_placement if best_placement else (0.0, 0.0, 0.0)

def place_all_greedy(n: int, strategy: str = 'spiral') -> Solution:
    """Place n trees using specified strategy."""
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    # Get initial positions based on strategy
    if strategy == 'hexagonal':
        positions = hexagonal_positions(n)
    elif strategy == 'spiral':
        positions = spiral_positions(n)
    elif strategy == 'concentric':
        positions = concentric_positions(n)
    else:  # grid
        positions = grid_positions(n)

    placements = []
    placed_polys = []

    for i, (px, py) in enumerate(positions):
        # Find valid rotation at this position
        if placed_polys:
            tree_index = STRtree(placed_polys)

            # Try original position with optimal rotation
            best_rot = find_best_rotation(px, py, placed_polys, tree_index)

            if best_rot is not None:
                placements.append((px, py, best_rot))
                placed_polys.append(make_tree_polygon(px, py, best_rot))
            else:
                # Position invalid, use radial placement
                x, y, rot = place_tree_radial(placed_polys, tree_index)
                placements.append((x, y, rot))
                placed_polys.append(make_tree_polygon(x, y, rot))
        else:
            rot = random.uniform(0, 360)
            placements.append((px, py, rot))
            placed_polys.append(make_tree_polygon(px, py, rot))

    return placements

# ============================================================================
# ADVANCED SIMULATED ANNEALING
# ============================================================================

def adaptive_sa(placements: Solution,
                iterations: int = 50000,
                temp_initial: float = 2.0,
                temp_final: float = 1e-6,
                verbose: bool = False) -> Solution:
    """Advanced simulated annealing with adaptive parameters."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_tree_polygon(x, y, d) for x, y, d in current]
    current_score = max(get_bounds(current)[2] - get_bounds(current)[0],
                        get_bounds(current)[3] - get_bounds(current)[1])

    best = list(current)
    best_score = current_score

    # Adaptive temperature schedule
    T = temp_initial
    cooling = (temp_final / temp_initial) ** (1.0 / iterations)

    # Adaptive move parameters
    max_shift = current_score * 0.15
    max_rotate = 45.0

    # Track acceptance for adaptation
    accepted = 0
    improved = 0
    window = 1000

    for it in range(iterations):
        # Pick random tree
        i = random.randrange(n)
        x, y, deg = current[i]

        # Adaptive move type based on progress
        progress = it / iterations

        if random.random() < 0.5 - 0.2 * progress:
            # Translation (more early, less late)
            new_x = x + random.gauss(0, max_shift)
            new_y = y + random.gauss(0, max_shift)
            new_deg = deg
        elif random.random() < 0.7:
            # Rotation
            new_x, new_y = x, y
            new_deg = (deg + random.gauss(0, max_rotate)) % 360
        else:
            # Combined small move
            new_x = x + random.gauss(0, max_shift * 0.5)
            new_y = y + random.gauss(0, max_shift * 0.5)
            new_deg = (deg + random.gauss(0, max_rotate * 0.5)) % 360

        # Create and validate new polygon
        new_poly = make_tree_polygon(new_x, new_y, new_deg)
        others = polys[:i] + polys[i+1:]

        if has_collision(new_poly, others):
            T *= cooling
            continue

        # Compute new score (bounding box)
        old_poly = polys[i]
        polys[i] = new_poly
        bounds = unary_union(polys).bounds
        new_score = max(bounds[2] - bounds[0], bounds[3] - bounds[1])

        # Acceptance criterion
        delta = new_score - current_score

        if delta <= 0:
            # Always accept improvements
            current[i] = (new_x, new_y, new_deg)
            current_score = new_score
            accepted += 1
            improved += 1

            if new_score < best_score:
                best_score = new_score
                best = list(current)
        elif T > 0 and random.random() < math.exp(-delta / T):
            # Probabilistic acceptance
            current[i] = (new_x, new_y, new_deg)
            current_score = new_score
            accepted += 1
        else:
            polys[i] = old_poly

        T *= cooling

        # Adaptive parameter adjustment every window iterations
        if (it + 1) % window == 0:
            accept_rate = accepted / window

            # Adjust move sizes based on acceptance rate
            if accept_rate < 0.2:
                max_shift *= 0.9
                max_rotate *= 0.9
            elif accept_rate > 0.4:
                max_shift *= 1.1
                max_rotate *= 1.1

            max_shift = max(0.001, min(max_shift, current_score * 0.3))
            max_rotate = max(1.0, min(max_rotate, 90.0))

            accepted = 0
            improved = 0

    return best

# ============================================================================
# COMPACTION OPTIMIZATION
# ============================================================================

def compact_solution(placements: Solution, iterations: int = 5000) -> Solution:
    """Compact solution by moving trees toward center."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_tree_polygon(x, y, d) for x, y, d in current]

    for _ in range(iterations):
        # Find current bounds and center
        bounds = unary_union(polys).bounds
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2

        # Pick a random tree
        i = random.randrange(n)
        x, y, deg = current[i]

        # Calculate direction toward center
        dx = cx - x
        dy = cy - y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.01:
            continue

        # Try to move toward center
        step = random.uniform(0.01, 0.1) * dist
        new_x = x + step * dx / dist
        new_y = y + step * dy / dist

        # Check collision
        new_poly = make_tree_polygon(new_x, new_y, deg)
        others = polys[:i] + polys[i+1:]

        if not has_collision(new_poly, others):
            # Check if this improves the bounding box
            old_poly = polys[i]
            polys[i] = new_poly

            new_bounds = unary_union(polys).bounds
            old_side = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
            new_side = max(new_bounds[2] - new_bounds[0], new_bounds[3] - new_bounds[1])

            if new_side <= old_side:
                current[i] = (new_x, new_y, deg)
            else:
                polys[i] = old_poly

    return current

def squeeze_boundaries(placements: Solution, iterations: int = 2000) -> Solution:
    """Move boundary trees inward."""
    n = len(placements)
    if n <= 2:
        return placements

    current = list(placements)
    polys = [make_tree_polygon(x, y, d) for x, y, d in current]

    for _ in range(iterations):
        bounds = unary_union(polys).bounds
        minx, miny, maxx, maxy = bounds
        side = max(maxx - minx, maxy - miny)

        # Find trees on boundaries
        boundary_trees = []
        for i, (x, y, d) in enumerate(current):
            poly = polys[i]
            pb = poly.bounds
            if (abs(pb[0] - minx) < 0.01 or abs(pb[2] - maxx) < 0.01 or
                abs(pb[1] - miny) < 0.01 or abs(pb[3] - maxy) < 0.01):
                boundary_trees.append(i)

        if not boundary_trees:
            break

        # Try to move a boundary tree inward
        i = random.choice(boundary_trees)
        x, y, deg = current[i]

        # Move toward center
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2

        step = random.uniform(0.01, 0.05)
        new_x = x + step * (cx - x)
        new_y = y + step * (cy - y)

        new_poly = make_tree_polygon(new_x, new_y, deg)
        others = polys[:i] + polys[i+1:]

        if not has_collision(new_poly, others):
            old_poly = polys[i]
            polys[i] = new_poly

            new_bounds = unary_union(polys).bounds
            new_side = max(new_bounds[2] - new_bounds[0], new_bounds[3] - new_bounds[1])

            if new_side < side:
                current[i] = (new_x, new_y, deg)
            else:
                polys[i] = old_poly

    return current

# ============================================================================
# LOCAL SEARCH REFINEMENT
# ============================================================================

def local_search(placements: Solution, iterations: int = 10000) -> Solution:
    """Fine-grained local search."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_tree_polygon(x, y, d) for x, y, d in current]
    current_score = max(get_bounds(current)[2] - get_bounds(current)[0],
                        get_bounds(current)[3] - get_bounds(current)[1])

    step_size = 0.02

    for _ in range(iterations):
        i = random.randrange(n)
        x, y, deg = current[i]

        # Try small moves in random direction
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        drot = random.gauss(0, 5)

        new_x = x + dx
        new_y = y + dy
        new_deg = (deg + drot) % 360

        new_poly = make_tree_polygon(new_x, new_y, new_deg)
        others = polys[:i] + polys[i+1:]

        if has_collision(new_poly, others):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        bounds = unary_union(polys).bounds
        new_score = max(bounds[2] - bounds[0], bounds[3] - bounds[1])

        if new_score < current_score:
            current[i] = (new_x, new_y, new_deg)
            current_score = new_score
        else:
            polys[i] = old_poly

    return current

# ============================================================================
# COORDINATE DESCENT
# ============================================================================

def coordinate_descent(placements: Solution, iterations: int = 3) -> Solution:
    """Optimize each tree's position independently."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)

    for _ in range(iterations):
        improved = False

        for i in range(n):
            polys = [make_tree_polygon(x, y, d) for x, y, d in current]
            current_score = max(get_bounds(current)[2] - get_bounds(current)[0],
                               get_bounds(current)[3] - get_bounds(current)[1])

            x, y, deg = current[i]
            others = polys[:i] + polys[i+1:]

            # Try small adjustments
            for dx, dy, drot in [
                (0.02, 0, 0), (-0.02, 0, 0),
                (0, 0.02, 0), (0, -0.02, 0),
                (0, 0, 10), (0, 0, -10),
                (0.01, 0.01, 5), (-0.01, -0.01, -5)
            ]:
                new_x = x + dx
                new_y = y + dy
                new_deg = (deg + drot) % 360

                new_poly = make_tree_polygon(new_x, new_y, new_deg)

                if has_collision(new_poly, others):
                    continue

                # Check improvement
                test_placements = current[:i] + [(new_x, new_y, new_deg)] + current[i+1:]
                new_score = get_bounding_square(test_placements)

                if new_score < current_score:
                    current[i] = (new_x, new_y, new_deg)
                    improved = True
                    break

        if not improved:
            break

    return current

# ============================================================================
# MULTI-STRATEGY OPTIMIZER
# ============================================================================

class AdvancedSolver:
    """Complete solver combining all optimization techniques."""

    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        random.seed(seed)
        np.random.seed(seed)

        self.solutions: Dict[int, Solution] = {}
        self.scores: Dict[int, float] = {}

    def optimize_single(self, n: int, max_attempts: int = 5) -> Solution:
        """Optimize for a single n using multiple strategies."""
        if n == 0:
            return []
        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            self.solutions[1] = sol
            self.scores[1] = get_bounding_square(sol)
            return sol

        best_solution = None
        best_score = float('inf')

        # Try different initialization strategies
        strategies = ['spiral', 'hexagonal', 'concentric']

        for attempt in range(max_attempts):
            strategy = strategies[attempt % len(strategies)]

            # Get starting solution
            if n - 1 in self.solutions and attempt == 0:
                # Build from previous solution
                prev = self.solutions[n - 1]
                prev_polys = [make_tree_polygon(x, y, d) for x, y, d in prev]
                tree_index = STRtree(prev_polys)

                x, y, rot = place_tree_radial(prev_polys, tree_index, n_attempts=80)
                solution = prev + [(x, y, rot)]
            else:
                # Fresh placement
                random.seed(self.seed + attempt * 1000 + n)
                solution = place_all_greedy(n, strategy)

            # Verify no overlaps
            overlaps = check_all_overlaps(solution)
            if overlaps:
                continue

            # Scale iterations based on n
            base_iter = 30000 + n * 200

            # Apply optimization pipeline
            solution = adaptive_sa(solution, iterations=base_iter)
            solution = compact_solution(solution, iterations=3000)
            solution = squeeze_boundaries(solution, iterations=2000)
            solution = local_search(solution, iterations=5000)
            solution = coordinate_descent(solution, iterations=3)

            # Final SA pass
            solution = adaptive_sa(solution, iterations=base_iter // 2,
                                  temp_initial=0.5, temp_final=1e-8)

            # Verify and score
            overlaps = check_all_overlaps(solution)
            if overlaps:
                continue

            score = get_bounding_square(solution)

            if score < best_score:
                best_score = score
                best_solution = solution

        if best_solution is None:
            # Fallback: simple incremental
            if n - 1 in self.solutions:
                prev = self.solutions[n - 1]
                prev_polys = [make_tree_polygon(x, y, d) for x, y, d in prev]
                tree_index = STRtree(prev_polys)
                x, y, rot = place_tree_radial(prev_polys, tree_index)
                best_solution = prev + [(x, y, rot)]
            else:
                best_solution = place_all_greedy(n, 'spiral')

        best_solution = center_solution(best_solution)
        self.solutions[n] = best_solution
        self.scores[n] = get_bounding_square(best_solution)

        return best_solution

    def solve_all(self, max_n: int = 200) -> Dict[int, Solution]:
        """Solve for all n from 1 to max_n."""
        start_time = time.time()

        if self.verbose:
            print(f"Advanced Solver: Solving n=1 to {max_n}")
            print("=" * 60)

        for n in range(1, max_n + 1):
            self.optimize_single(n)

            if self.verbose and (n % 10 == 0 or n <= 5):
                elapsed = time.time() - start_time
                score = self.compute_total_score()
                print(f"n={n:3d}: side={self.scores[n]:.4f}, "
                      f"cumulative_score={score:.2f}, time={elapsed:.1f}s")

        if self.verbose:
            print("=" * 60)
            print(f"Final Score: {self.compute_total_score():.4f}")
            print(f"Total Time: {time.time() - start_time:.1f}s")

        return self.solutions

    def compute_total_score(self) -> float:
        """Compute competition score: sum of (side^2 / n)."""
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, get_bounding_square(sol))
            total += (side ** 2) / n
        return total

# ============================================================================
# SUBMISSION CREATION
# ============================================================================

def create_submission(solutions: Dict[int, Solution],
                     output_path: str = "submission.csv") -> str:
    """Create submission CSV in exact Kaggle format."""
    with open(output_path, "w") as f:
        f.write("id,x,y,deg\n")

        for n in range(1, 201):
            if n not in solutions:
                raise ValueError(f"Missing solution for n={n}")

            positions = solutions[n]
            if len(positions) != n:
                raise ValueError(f"Wrong count for n={n}: got {len(positions)}")

            # Get bounds to normalize coordinates
            polys = [make_tree_polygon(x, y, d) for x, y, d in positions]
            bounds = unary_union(polys).bounds
            min_x, min_y = bounds[0], bounds[1]

            for idx, (x, y, deg) in enumerate(positions):
                # Shift coordinates so min is at 0
                X = x - min_x
                Y = y - min_y
                f.write(f"{n:03d}_{idx},s{X:.6f},s{Y:.6f},s{deg:.6f}\n")

    return output_path

def validate_submission(solutions: Dict[int, Solution]) -> Tuple[bool, List[str]]:
    """Validate all solutions."""
    errors = []

    for n in range(1, 201):
        if n not in solutions:
            errors.append(f"Missing n={n}")
            continue

        if len(solutions[n]) != n:
            errors.append(f"n={n}: wrong count {len(solutions[n])}")
            continue

        overlaps = check_all_overlaps(solutions[n])
        if overlaps:
            errors.append(f"n={n}: {len(overlaps)} overlapping pairs")

    return len(errors) == 0, errors

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete optimization."""
    print("=" * 70)
    print("SANTA 2025 - ADVANCED TREE PACKING SOLVER")
    print("=" * 70)
    print()

    # Run multiple seeds and keep best
    best_solutions = None
    best_total_score = float('inf')

    seeds = [42, 123, 456, 789, 2025]

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Running with seed={seed}")
        print(f"{'='*70}")

        solver = AdvancedSolver(seed=seed, verbose=True)
        solutions = solver.solve_all(max_n=200)

        # Validate
        valid, errors = validate_submission(solutions)
        if not valid:
            print(f"Validation errors: {errors[:5]}")
            continue

        score = solver.compute_total_score()
        print(f"Seed {seed} Score: {score:.4f}")

        if score < best_total_score:
            best_total_score = score
            best_solutions = solutions

    if best_solutions is None:
        print("ERROR: No valid solutions found!")
        return

    # Create submission
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Best Score: {best_total_score:.4f}")

    output_path = create_submission(best_solutions, "submission.csv")
    print(f"Submission saved to: {output_path}")

    # Final validation
    valid, errors = validate_submission(best_solutions)
    if valid:
        print("✓ All solutions validated successfully!")
    else:
        print(f"✗ Validation errors: {errors}")

if __name__ == "__main__":
    main()
