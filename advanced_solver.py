#!/usr/bin/env python3
"""
advanced_solver.py - Advanced Tree Packing Solver for Santa 2025

Implements state-of-the-art optimization techniques:
1. Genetic Algorithm (GA) for global optimization
2. Basin Hopping for escaping local minima
3. Extended Simulated Annealing with reheat
4. Multi-stage compaction
5. Rotation optimization
6. Multi-seed ensemble

Target: Score < 60

Run: python advanced_solver.py
Output: submission.csv
"""

import math
import random
import time
import copy
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from functools import lru_cache

from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Solver configuration - tuned for best quality."""
    # Genetic Algorithm
    ga_population_size: int = 20
    ga_generations: int = 30
    ga_mutation_rate: float = 0.3
    ga_crossover_rate: float = 0.7
    ga_elite_size: int = 3

    # Basin Hopping
    bh_iterations: int = 15
    bh_temperature: float = 1.0
    bh_step_size: float = 0.3

    # Simulated Annealing (very long runs)
    sa_iterations_base: int = 25000
    sa_iterations_scale: float = 200  # Additional iterations per n
    sa_temp_initial: float = 3.0
    sa_temp_final: float = 1e-10
    sa_reheat_interval: int = 8000
    sa_reheat_factor: float = 0.25

    # Compaction
    compact_iterations: int = 8000
    squeeze_iterations: int = 4000

    # Local Search
    local_iterations: int = 10000

    # Rotation Optimization
    rotation_iterations: int = 5000

    # Multi-restart
    num_restarts: int = 5

    # Seeds for ensemble
    seeds: List[int] = None

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456, 789, 2025, 1337, 7777, 9999]

# Global config
CFG = Config()

# ============================================================================
# TREE GEOMETRY
# ============================================================================

# Official tree polygon (15 vertices)
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
TREE_AREA = BASE_POLYGON.area

# Type aliases
Placement = Tuple[float, float, float]  # (x, y, angle_degrees)
Solution = List[Placement]

# ============================================================================
# CORE GEOMETRY FUNCTIONS
# ============================================================================

def make_polygon(x: float, y: float, angle: float) -> Polygon:
    """Create tree polygon at position with rotation."""
    poly = BASE_POLYGON
    if angle != 0:
        poly = affinity.rotate(poly, angle, origin=(0, 0))
    if x != 0 or y != 0:
        poly = affinity.translate(poly, xoff=x, yoff=y)
    return poly

def check_collision(poly: Polygon, others: List[Polygon]) -> bool:
    """Check if polygon collides with any other (touching is OK)."""
    for other in others:
        if poly.intersects(other) and not poly.touches(other):
            return True
    return False

def check_collision_indexed(poly: Polygon, index: STRtree, all_polys: List[Polygon]) -> bool:
    """Fast collision check using spatial index."""
    candidates = index.query(poly)
    for i in candidates:
        if poly.intersects(all_polys[i]) and not poly.touches(all_polys[i]):
            return True
    return False

def get_bounding_side(placements: Solution) -> float:
    """Get side length of bounding square."""
    if not placements:
        return 0.0
    polys = [make_polygon(x, y, d) for x, y, d in placements]
    bounds = unary_union(polys).bounds
    return max(bounds[2] - bounds[0], bounds[3] - bounds[1])

def get_bounding_side_polys(polys: List[Polygon]) -> float:
    """Get side from polygon list."""
    if not polys:
        return 0.0
    bounds = unary_union(polys).bounds
    return max(bounds[2] - bounds[0], bounds[3] - bounds[1])

def center_solution(placements: Solution) -> Solution:
    """Center solution around origin."""
    if not placements:
        return placements
    polys = [make_polygon(x, y, d) for x, y, d in placements]
    bounds = unary_union(polys).bounds
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]

def count_overlaps(placements: Solution) -> int:
    """Count overlapping tree pairs."""
    polys = [make_polygon(x, y, d) for x, y, d in placements]
    count = 0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                count += 1
    return count

def is_valid(placements: Solution) -> bool:
    """Check if solution has no overlaps."""
    return count_overlaps(placements) == 0

# ============================================================================
# INITIAL PLACEMENT STRATEGIES
# ============================================================================

def fermat_spiral(n: int, scale: float = 0.48) -> List[Tuple[float, float]]:
    """Generate positions using Fermat (golden angle) spiral."""
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    positions = []
    for i in range(n):
        r = scale * math.sqrt(i)
        theta = i * golden_angle
        positions.append((r * math.cos(theta), r * math.sin(theta)))
    return positions

def hexagonal_grid(n: int, spacing: float = 0.72) -> List[Tuple[float, float]]:
    """Generate hexagonal grid positions."""
    positions = [(0.0, 0.0)]
    if n <= 1:
        return positions[:n]

    dx = spacing * TREE_WIDTH
    dy = spacing * TREE_HEIGHT * 0.866

    ring = 1
    while len(positions) < n:
        for side in range(6):
            for step in range(ring):
                if len(positions) >= n:
                    break
                angle = math.pi / 3 * side
                start_x = ring * dx * math.cos(angle - math.pi / 6)
                start_y = ring * dy * math.sin(angle - math.pi / 6)
                side_angle = angle + math.pi / 2
                x = start_x + step * dx * math.cos(side_angle) * 0.5
                y = start_y + step * dy * math.sin(side_angle) * 0.5
                positions.append((x, y))
        ring += 1

    return positions[:n]

def concentric_rings(n: int, spacing: float = 0.75) -> List[Tuple[float, float]]:
    """Generate concentric circle positions."""
    positions = [(0.0, 0.0)]
    if n <= 1:
        return positions[:n]

    ring = 1
    base_radius = spacing * max(TREE_WIDTH, TREE_HEIGHT)

    while len(positions) < n:
        radius = ring * base_radius
        circumference = 2 * math.pi * radius
        trees_in_ring = max(6, int(circumference / (spacing * TREE_WIDTH)))

        for i in range(trees_in_ring):
            if len(positions) >= n:
                break
            angle = 2 * math.pi * i / trees_in_ring + (ring * 0.1)  # Offset each ring
            positions.append((radius * math.cos(angle), radius * math.sin(angle)))
        ring += 1

    return positions[:n]

# ============================================================================
# TREE PLACEMENT
# ============================================================================

def find_valid_rotation(x: float, y: float, existing_polys: List[Polygon],
                       index: Optional[STRtree] = None) -> Optional[float]:
    """Find a valid rotation angle at given position."""
    for angle in range(0, 360, 10):
        poly = make_polygon(x, y, angle)
        if index:
            if not check_collision_indexed(poly, index, existing_polys):
                return float(angle)
        else:
            if not check_collision(poly, existing_polys):
                return float(angle)
    return None

def place_tree_radially(existing_polys: List[Polygon],
                       index: Optional[STRtree],
                       attempts: int = 60) -> Placement:
    """Place tree using radial approach with binary search."""
    if not existing_polys:
        return (0.0, 0.0, random.uniform(0, 360))

    best_placement = None
    best_radius = float('inf')

    for _ in range(attempts):
        # Weighted angle favoring diagonals
        while True:
            theta = random.uniform(0, 2 * math.pi)
            if random.random() < abs(math.sin(2 * theta)) + 0.15:
                break

        vx, vy = math.cos(theta), math.sin(theta)

        # Binary search for minimum radius
        low, high = 0.0, 15.0
        best_r = high

        while high - low > 0.02:
            mid = (low + high) / 2
            px, py = mid * vx, mid * vy

            valid = False
            for rot in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
                poly = make_polygon(px, py, rot)
                if index:
                    if not check_collision_indexed(poly, index, existing_polys):
                        valid = True
                        break
                else:
                    if not check_collision(poly, existing_polys):
                        valid = True
                        break

            if valid:
                high = mid
                best_r = mid
            else:
                low = mid

        if best_r < best_radius:
            px, py = best_r * vx, best_r * vy
            best_rot = 0.0
            for rot in range(0, 360, 10):
                poly = make_polygon(px, py, rot)
                if index:
                    if not check_collision_indexed(poly, index, existing_polys):
                        best_rot = float(rot)
                        break
                else:
                    if not check_collision(poly, existing_polys):
                        best_rot = float(rot)
                        break

            best_radius = best_r
            best_placement = (px, py, best_rot)

    return best_placement if best_placement else (10.0, 0.0, 0.0)

def build_initial_solution(n: int, strategy: str = 'spiral', scale: float = 0.48) -> Solution:
    """Build initial solution using specified strategy."""
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    # Get positions based on strategy
    if strategy == 'spiral':
        positions = fermat_spiral(n, scale)
    elif strategy == 'hex':
        positions = hexagonal_grid(n, scale + 0.2)
    else:  # concentric
        positions = concentric_rings(n, scale + 0.25)

    placements = []
    placed_polys = []

    for px, py in positions:
        if not placed_polys:
            angle = random.uniform(0, 360)
            placements.append((px, py, angle))
            placed_polys.append(make_polygon(px, py, angle))
        else:
            index = STRtree(placed_polys)
            valid_angle = find_valid_rotation(px, py, placed_polys, index)

            if valid_angle is not None:
                placements.append((px, py, valid_angle))
                placed_polys.append(make_polygon(px, py, valid_angle))
            else:
                # Fall back to radial placement
                placement = place_tree_radially(placed_polys, index)
                placements.append(placement)
                placed_polys.append(make_polygon(*placement))

    return placements

# ============================================================================
# SIMULATED ANNEALING (Extended with Reheat)
# ============================================================================

def extended_simulated_annealing(placements: Solution,
                                  iterations: int,
                                  temp_start: float = 3.0,
                                  temp_end: float = 1e-10,
                                  reheat_interval: int = 8000,
                                  reheat_factor: float = 0.25) -> Solution:
    """Extended SA with periodic reheating for escaping local minima."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_polygon(x, y, d) for x, y, d in current]
    current_side = get_bounding_side_polys(polys)

    best = list(current)
    best_side = current_side

    T = temp_start
    base_cooling = (temp_end / temp_start) ** (1.0 / iterations)

    shift_size = current_side * 0.12
    rotate_size = 45.0

    last_improvement = 0
    stagnation_count = 0

    for it in range(iterations):
        # Periodic reheat
        if it > 0 and it % reheat_interval == 0:
            T = max(T, temp_start * reheat_factor * (1 - it / iterations))
            shift_size = best_side * 0.1
            rotate_size = 40.0
            stagnation_count = 0

        # Select random tree
        i = random.randrange(n)
        x, y, deg = current[i]

        # Adaptive move type based on progress
        progress = it / iterations
        r = random.random()

        if r < 0.45:
            # Translation
            new_x = x + random.gauss(0, shift_size)
            new_y = y + random.gauss(0, shift_size)
            new_deg = deg
        elif r < 0.75:
            # Rotation
            new_x, new_y = x, y
            new_deg = (deg + random.gauss(0, rotate_size)) % 360
        elif r < 0.92:
            # Combined move
            new_x = x + random.gauss(0, shift_size * 0.6)
            new_y = y + random.gauss(0, shift_size * 0.6)
            new_deg = (deg + random.gauss(0, rotate_size * 0.6)) % 360
        else:
            # Large exploration jump
            new_x = x + random.gauss(0, shift_size * 2.5)
            new_y = y + random.gauss(0, shift_size * 2.5)
            new_deg = random.uniform(0, 360)

        # Create candidate and check collision
        new_poly = make_polygon(new_x, new_y, new_deg)
        other_polys = polys[:i] + polys[i+1:]

        if check_collision(new_poly, other_polys):
            T *= base_cooling
            stagnation_count += 1
            continue

        # Evaluate
        old_poly = polys[i]
        polys[i] = new_poly
        new_side = get_bounding_side_polys(polys)

        delta = new_side - current_side

        # Acceptance decision
        accept = False
        if delta <= 0:
            accept = True
        elif T > 0 and random.random() < math.exp(-delta / T):
            accept = True

        if accept:
            current[i] = (new_x, new_y, new_deg)
            current_side = new_side
            stagnation_count = 0

            if new_side < best_side:
                best_side = new_side
                best = list(current)
                last_improvement = it
        else:
            polys[i] = old_poly

        T *= base_cooling

        # Adaptive parameter adjustment
        if stagnation_count > 500:
            shift_size = max(0.003, shift_size * 0.95)
            rotate_size = max(1.0, rotate_size * 0.95)
            stagnation_count = 0

    return best

# ============================================================================
# GENETIC ALGORITHM
# ============================================================================

def crossover(parent1: Solution, parent2: Solution) -> Solution:
    """Single-point crossover between two solutions."""
    n = len(parent1)
    if n <= 2:
        return list(parent1)

    # Choose crossover point
    point = random.randint(1, n - 1)

    # Create child by taking positions from parent1, angles from mix
    child = []
    for i in range(n):
        if i < point:
            child.append(parent1[i])
        else:
            # Take position from parent1 but try angle from parent2
            x, y, _ = parent1[i]
            _, _, deg = parent2[i]
            child.append((x, y, deg))

    return child

def mutate(solution: Solution, mutation_rate: float = 0.3) -> Solution:
    """Mutate solution by perturbing random trees."""
    n = len(solution)
    mutated = list(solution)
    polys = [make_polygon(x, y, d) for x, y, d in mutated]
    current_side = get_bounding_side_polys(polys)

    for i in range(n):
        if random.random() < mutation_rate:
            x, y, deg = mutated[i]

            # Random perturbation
            new_x = x + random.gauss(0, current_side * 0.05)
            new_y = y + random.gauss(0, current_side * 0.05)
            new_deg = (deg + random.gauss(0, 20)) % 360

            new_poly = make_polygon(new_x, new_y, new_deg)
            others = polys[:i] + polys[i+1:]

            if not check_collision(new_poly, others):
                mutated[i] = (new_x, new_y, new_deg)
                polys[i] = new_poly

    return mutated

def genetic_algorithm(initial_solution: Solution,
                     population_size: int = 20,
                     generations: int = 30,
                     mutation_rate: float = 0.3,
                     crossover_rate: float = 0.7,
                     elite_size: int = 3) -> Solution:
    """Genetic algorithm for global optimization."""
    n = len(initial_solution)
    if n <= 2:
        return initial_solution

    # Initialize population with variations of initial solution
    population = [initial_solution]
    for _ in range(population_size - 1):
        variant = mutate(list(initial_solution), mutation_rate=0.5)
        if is_valid(variant):
            population.append(variant)
        else:
            population.append(list(initial_solution))

    best_solution = initial_solution
    best_score = get_bounding_side(initial_solution)

    for gen in range(generations):
        # Evaluate fitness (lower is better)
        fitness = []
        for sol in population:
            if is_valid(sol):
                fitness.append(get_bounding_side(sol))
            else:
                fitness.append(float('inf'))

        # Sort by fitness
        sorted_pop = [x for _, x in sorted(zip(fitness, population), key=lambda p: p[0])]
        sorted_fitness = sorted(fitness)

        # Update best
        if sorted_fitness[0] < best_score:
            best_score = sorted_fitness[0]
            best_solution = list(sorted_pop[0])

        # Selection (elitism + tournament)
        new_population = sorted_pop[:elite_size]  # Keep elites

        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            candidates = random.sample(list(zip(sorted_fitness, sorted_pop)), tournament_size)
            parent1 = min(candidates, key=lambda x: x[0])[1]
            candidates = random.sample(list(zip(sorted_fitness, sorted_pop)), tournament_size)
            parent2 = min(candidates, key=lambda x: x[0])[1]

            # Crossover
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = list(parent1)

            # Mutation
            child = mutate(child, mutation_rate)

            # Validate and add
            if is_valid(child):
                new_population.append(child)
            else:
                new_population.append(list(parent1))

        population = new_population

    return best_solution

# ============================================================================
# BASIN HOPPING
# ============================================================================

def local_minimize(solution: Solution, iterations: int = 2000) -> Solution:
    """Local minimization using gradient-free descent."""
    return extended_simulated_annealing(
        solution,
        iterations=iterations,
        temp_start=0.5,
        temp_end=1e-8,
        reheat_interval=iterations + 1  # No reheat
    )

def basin_hopping(initial_solution: Solution,
                  n_iterations: int = 15,
                  temperature: float = 1.0,
                  step_size: float = 0.3) -> Solution:
    """Basin hopping for global optimization."""
    n = len(initial_solution)
    if n <= 2:
        return initial_solution

    current = local_minimize(initial_solution, iterations=3000)
    current_score = get_bounding_side(current)

    best = list(current)
    best_score = current_score

    for iteration in range(n_iterations):
        # Random perturbation (hop)
        candidate = list(current)
        polys = [make_polygon(x, y, d) for x, y, d in candidate]

        for i in range(n):
            if random.random() < step_size:
                x, y, deg = candidate[i]
                new_x = x + random.gauss(0, current_score * 0.15)
                new_y = y + random.gauss(0, current_score * 0.15)
                new_deg = (deg + random.gauss(0, 30)) % 360

                new_poly = make_polygon(new_x, new_y, new_deg)
                others = polys[:i] + polys[i+1:]

                if not check_collision(new_poly, others):
                    candidate[i] = (new_x, new_y, new_deg)
                    polys[i] = new_poly

        # Local minimization
        candidate = local_minimize(candidate, iterations=3000)
        candidate_score = get_bounding_side(candidate)

        # Metropolis acceptance
        delta = candidate_score - current_score

        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current = candidate
            current_score = candidate_score

            if candidate_score < best_score:
                best_score = candidate_score
                best = list(candidate)

    return best

# ============================================================================
# COMPACTION ALGORITHMS
# ============================================================================

def compact_toward_center(placements: Solution, iterations: int) -> Solution:
    """Move trees toward center to reduce bounding box."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_polygon(x, y, d) for x, y, d in current]
    current_side = get_bounding_side_polys(polys)

    for _ in range(iterations):
        bounds = unary_union(polys).bounds
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2

        i = random.randrange(n)
        x, y, deg = current[i]

        # Direction toward center
        dx, dy = cx - x, cy - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.01:
            continue

        # Random step toward center
        step = random.uniform(0.003, min(0.08, dist * 0.4))
        new_x = x + step * dx / dist
        new_y = y + step * dy / dist
        new_deg = deg

        # Occasionally adjust rotation too
        if random.random() < 0.25:
            new_deg = (deg + random.gauss(0, 10)) % 360

        new_poly = make_polygon(new_x, new_y, new_deg)
        other_polys = polys[:i] + polys[i+1:]

        if check_collision(new_poly, other_polys):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_side = get_bounding_side_polys(polys)

        if new_side <= current_side:
            current[i] = (new_x, new_y, new_deg)
            current_side = new_side
        else:
            polys[i] = old_poly

    return current

def squeeze_boundaries(placements: Solution, iterations: int) -> Solution:
    """Move boundary trees inward."""
    n = len(placements)
    if n <= 2:
        return placements

    current = list(placements)
    polys = [make_polygon(x, y, d) for x, y, d in current]

    for _ in range(iterations):
        bounds = unary_union(polys).bounds
        minx, miny, maxx, maxy = bounds
        side = max(maxx - minx, maxy - miny)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

        # Find trees on boundary
        boundary_trees = []
        for i, (x, y, d) in enumerate(current):
            pb = polys[i].bounds
            on_boundary = (abs(pb[0] - minx) < 0.03 or abs(pb[2] - maxx) < 0.03 or
                          abs(pb[1] - miny) < 0.03 or abs(pb[3] - maxy) < 0.03)
            if on_boundary:
                boundary_trees.append(i)

        if not boundary_trees:
            break

        i = random.choice(boundary_trees)
        x, y, deg = current[i]

        # Move toward center
        dx, dy = cx - x, cy - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.01:
            continue

        step = random.uniform(0.005, 0.03)
        new_x = x + step * dx / dist
        new_y = y + step * dy / dist

        new_poly = make_polygon(new_x, new_y, deg)
        other_polys = polys[:i] + polys[i+1:]

        if check_collision(new_poly, other_polys):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_side = get_bounding_side_polys(polys)

        if new_side < side:
            current[i] = (new_x, new_y, deg)
        else:
            polys[i] = old_poly

    return current

def optimize_rotations(placements: Solution, iterations: int) -> Solution:
    """Fine-tune rotations for better packing."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_polygon(x, y, d) for x, y, d in current]
    current_side = get_bounding_side_polys(polys)

    for _ in range(iterations):
        i = random.randrange(n)
        x, y, deg = current[i]

        new_deg = (deg + random.gauss(0, 25)) % 360

        new_poly = make_polygon(x, y, new_deg)
        other_polys = polys[:i] + polys[i+1:]

        if check_collision(new_poly, other_polys):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_side = get_bounding_side_polys(polys)

        if new_side < current_side:
            current[i] = (x, y, new_deg)
            current_side = new_side
        else:
            polys[i] = old_poly

    return current

def local_refinement(placements: Solution, iterations: int) -> Solution:
    """Fine-grained local search."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_polygon(x, y, d) for x, y, d in current]
    current_side = get_bounding_side_polys(polys)

    step = 0.01
    rot_step = 5.0

    for it in range(iterations):
        i = random.randrange(n)
        x, y, deg = current[i]

        new_x = x + random.gauss(0, step)
        new_y = y + random.gauss(0, step)
        new_deg = (deg + random.gauss(0, rot_step)) % 360

        new_poly = make_polygon(new_x, new_y, new_deg)
        other_polys = polys[:i] + polys[i+1:]

        if check_collision(new_poly, other_polys):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_side = get_bounding_side_polys(polys)

        if new_side < current_side:
            current[i] = (new_x, new_y, new_deg)
            current_side = new_side
        else:
            polys[i] = old_poly

        # Decay step size
        if it % 2000 == 0 and it > 0:
            step = max(0.002, step * 0.85)
            rot_step = max(1.0, rot_step * 0.85)

    return current

# ============================================================================
# MAIN SOLVER
# ============================================================================

class AdvancedSolver:
    """Complete solver with all advanced optimization techniques."""

    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        random.seed(seed)

        self.solutions: Dict[int, Solution] = {}
        self.scores: Dict[int, float] = {}

    def optimize_single(self, n: int) -> Solution:
        """Optimize solution for n trees using all techniques."""
        if n == 0:
            return []

        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            self.solutions[1] = sol
            self.scores[1] = get_bounding_side(sol)
            return sol

        best_solution = None
        best_side = float('inf')

        # Try multiple restarts with different strategies
        strategies = ['spiral', 'hex', 'concentric']
        scales = [0.46, 0.48, 0.50, 0.52]

        for restart in range(CFG.num_restarts):
            strategy = strategies[restart % len(strategies)]
            scale = scales[restart % len(scales)]

            # Build initial solution
            if n - 1 in self.solutions and restart == 0:
                # Extend from previous
                prev = self.solutions[n - 1]
                prev_polys = [make_polygon(x, y, d) for x, y, d in prev]
                index = STRtree(prev_polys)
                new_tree = place_tree_radially(prev_polys, index, attempts=80)
                sol = prev + [new_tree]
            else:
                random.seed(self.seed + n * 1000 + restart * 100)
                sol = build_initial_solution(n, strategy, scale)

            if not is_valid(sol):
                sol = build_initial_solution(n, 'spiral', 0.55)
                if not is_valid(sol):
                    continue

            # Calculate iterations based on n
            sa_iters = CFG.sa_iterations_base + int(n * CFG.sa_iterations_scale)

            # Phase 1: Extended Simulated Annealing
            sol = extended_simulated_annealing(
                sol,
                iterations=sa_iters,
                temp_start=CFG.sa_temp_initial,
                temp_end=CFG.sa_temp_final,
                reheat_interval=CFG.sa_reheat_interval,
                reheat_factor=CFG.sa_reheat_factor
            )

            # Phase 2: Genetic Algorithm (for larger n)
            if n >= 10:
                sol = genetic_algorithm(
                    sol,
                    population_size=CFG.ga_population_size,
                    generations=CFG.ga_generations,
                    mutation_rate=CFG.ga_mutation_rate,
                    crossover_rate=CFG.ga_crossover_rate,
                    elite_size=CFG.ga_elite_size
                )

            # Phase 3: Basin Hopping
            if n >= 5:
                sol = basin_hopping(
                    sol,
                    n_iterations=CFG.bh_iterations,
                    temperature=CFG.bh_temperature,
                    step_size=CFG.bh_step_size
                )

            # Phase 4: Compaction
            sol = compact_toward_center(sol, CFG.compact_iterations)
            sol = squeeze_boundaries(sol, CFG.squeeze_iterations)

            # Phase 5: Rotation optimization
            sol = optimize_rotations(sol, CFG.rotation_iterations)

            # Phase 6: Local refinement
            sol = local_refinement(sol, CFG.local_iterations)

            # Phase 7: Final SA pass
            sol = extended_simulated_annealing(
                sol,
                iterations=sa_iters // 2,
                temp_start=1.0,
                temp_end=1e-10,
                reheat_interval=sa_iters,  # No reheat
                reheat_factor=0
            )

            # Phase 8: Final compaction
            sol = compact_toward_center(sol, CFG.compact_iterations // 2)

            # Validate and score
            if not is_valid(sol):
                continue

            side = get_bounding_side(sol)
            if side < best_side:
                best_side = side
                best_solution = sol

        # Fallback
        if best_solution is None:
            if n - 1 in self.solutions:
                prev = self.solutions[n - 1]
                prev_polys = [make_polygon(x, y, d) for x, y, d in prev]
                index = STRtree(prev_polys)
                new_tree = place_tree_radially(prev_polys, index)
                best_solution = prev + [new_tree]
            else:
                best_solution = build_initial_solution(n, 'spiral', 0.6)

        best_solution = center_solution(best_solution)
        self.solutions[n] = best_solution
        self.scores[n] = get_bounding_side(best_solution)

        return best_solution

    def solve_all(self, max_n: int = 200) -> Dict[int, Solution]:
        """Solve for all n from 1 to max_n."""
        start_time = time.time()

        if self.verbose:
            print(f"Advanced Solver: n=1 to {max_n}")
            print(f"Config: SA={CFG.sa_iterations_base}+, GA={CFG.ga_generations}gen, BH={CFG.bh_iterations}iter")
            print("=" * 70)

        for n in range(1, max_n + 1):
            self.optimize_single(n)

            if self.verbose and (n % 10 == 0 or n <= 5):
                elapsed = time.time() - start_time
                total_score = self.compute_total_score()
                eta = (elapsed / n) * (max_n - n) if n > 0 else 0
                print(f"n={n:3d}: side={self.scores[n]:.4f}, "
                      f"score={total_score:.2f}, "
                      f"time={elapsed:.0f}s, ETA={eta:.0f}s")

        if self.verbose:
            print("=" * 70)
            print(f"FINAL SCORE: {self.compute_total_score():.4f}")
            print(f"Total time: {time.time() - start_time:.0f}s")

        return self.solutions

    def compute_total_score(self) -> float:
        """Compute competition score: sum(side^2 / n)."""
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, get_bounding_side(sol))
            total += (side ** 2) / n
        return total

# ============================================================================
# SUBMISSION FILE CREATION
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
                raise ValueError(f"n={n}: expected {n} trees, got {len(positions)}")

            # Normalize coordinates
            polys = [make_polygon(x, y, d) for x, y, d in positions]
            bounds = unary_union(polys).bounds
            min_x, min_y = bounds[0], bounds[1]

            for idx, (x, y, deg) in enumerate(positions):
                X = x - min_x
                Y = y - min_y
                f.write(f"{n:03d}_{idx},s{X:.6f},s{Y:.6f},s{deg:.6f}\n")

    return output_path

def validate_solutions(solutions: Dict[int, Solution]) -> Tuple[bool, List[str]]:
    """Validate all solutions."""
    errors = []

    for n in range(1, 201):
        if n not in solutions:
            errors.append(f"Missing n={n}")
            continue

        if len(solutions[n]) != n:
            errors.append(f"n={n}: wrong tree count")
            continue

        overlaps = count_overlaps(solutions[n])
        if overlaps > 0:
            errors.append(f"n={n}: {overlaps} overlapping pairs")

    return len(errors) == 0, errors

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the complete advanced optimization pipeline."""
    print("=" * 70)
    print("ADVANCED TREE PACKING SOLVER")
    print("Genetic Algorithm + Basin Hopping + Extended SA")
    print("Target Score: < 60")
    print("=" * 70)
    print()

    best_overall_score = float('inf')
    best_overall_solutions = None

    # Run with multiple seeds
    for seed in CFG.seeds:
        print(f"\n{'='*70}")
        print(f"Running with seed = {seed}")
        print(f"{'='*70}")

        solver = AdvancedSolver(seed=seed, verbose=True)
        solutions = solver.solve_all(max_n=200)

        # Validate
        valid, errors = validate_solutions(solutions)
        if not valid:
            print(f"Validation failed: {errors[:3]}")
            continue

        score = solver.compute_total_score()
        print(f"\nSeed {seed} final score: {score:.4f}")

        if score < best_overall_score:
            best_overall_score = score
            best_overall_solutions = copy.deepcopy(solutions)
            print(f"*** NEW BEST SCORE: {score:.4f} ***")

            # Save intermediate result
            create_submission(best_overall_solutions, "submission.csv")

    # Output final results
    if best_overall_solutions is None:
        print("\nERROR: No valid solutions found!")
        return

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Best Score: {best_overall_score:.4f}")

    # Create final submission
    output_path = create_submission(best_overall_solutions, "submission.csv")
    print(f"Submission saved to: {output_path}")

    # Final validation
    valid, errors = validate_solutions(best_overall_solutions)
    if valid:
        print("✓ All 200 solutions validated successfully!")
    else:
        print(f"✗ Validation errors: {errors}")

    # Score breakdown
    print(f"\nScore Analysis:")
    sides = {n: get_bounding_side(best_overall_solutions[n]) for n in range(1, 201)}
    contributions = {n: (sides[n] ** 2) / n for n in range(1, 201)}

    worst = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 worst contributors:")
    for n, contrib in worst:
        print(f"  n={n:3d}: side={sides[n]:.4f}, contribution={contrib:.4f}")

    print(f"\nSide range: {min(sides.values()):.4f} - {max(sides.values()):.4f}")

if __name__ == "__main__":
    main()
