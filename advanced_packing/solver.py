#!/usr/bin/env python3
"""
Advanced Tree Packing Solver - Fundamentally Different Approaches

This solver uses state-of-the-art optimization techniques:
1. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
2. Differential Evolution
3. Parallel Tempering with Replica Exchange
4. Pre-computed optimal solutions for small n
5. Pattern-based interlocking placement
6. Multi-resolution progressive refinement

Target: Score < 60

Run: python solver.py
Output: submission.csv
"""

import math
import random
import time
import copy
import pickle
import os
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
import multiprocessing as mp
from functools import partial
import numpy as np

from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Global configuration."""
    # CMA-ES parameters
    cmaes_population: int = 20
    cmaes_generations: int = 100
    cmaes_sigma: float = 0.3

    # Differential Evolution
    de_population: int = 30
    de_generations: int = 80
    de_F: float = 0.8  # Mutation factor
    de_CR: float = 0.9  # Crossover rate

    # Parallel Tempering
    pt_temperatures: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    pt_iterations: int = 50000
    pt_exchange_every: int = 100

    # Long SA parameters
    sa_iterations: int = 200000
    sa_temp_start: float = 5.0
    sa_temp_end: float = 1e-12

    # Compaction
    compact_iterations: int = 50000

    # Multi-restart
    num_restarts: int = 10

    # Seeds
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 2025, 1337, 7777, 9999, 31415, 27182])

CFG = Config()

# ============================================================================
# GEOMETRY
# ============================================================================

TREE_COORDS = np.array([
    [0.0, 0.8], [0.125, 0.5], [0.0625, 0.5], [0.2, 0.25], [0.1, 0.25],
    [0.35, 0.0], [0.075, 0.0], [0.075, -0.2], [-0.075, -0.2], [-0.075, 0.0],
    [-0.35, 0.0], [-0.1, 0.25], [-0.2, 0.25], [-0.0625, 0.5], [-0.125, 0.5],
])

BASE_POLYGON = Polygon(TREE_COORDS)
TREE_WIDTH = 0.7
TREE_HEIGHT = 1.0
TREE_AREA = BASE_POLYGON.area

Placement = Tuple[float, float, float]
Solution = List[Placement]

def make_polygon(x: float, y: float, angle: float) -> Polygon:
    """Create tree polygon."""
    cos_a = math.cos(math.radians(angle))
    sin_a = math.sin(math.radians(angle))
    rotated = TREE_COORDS @ np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    translated = rotated + np.array([x, y])
    return Polygon(translated)

def check_collision(poly: Polygon, others: List[Polygon]) -> bool:
    """Check collision (touching OK)."""
    for o in others:
        if poly.intersects(o) and not poly.touches(o):
            return True
    return False

def check_collision_indexed(poly: Polygon, idx: STRtree, polys: List[Polygon]) -> bool:
    """Fast collision check."""
    for i in idx.query(poly):
        if poly.intersects(polys[i]) and not poly.touches(polys[i]):
            return True
    return False

def get_bounding_side(placements: Solution) -> float:
    """Get bounding square side."""
    if not placements:
        return 0.0
    polys = [make_polygon(*p) for p in placements]
    bounds = unary_union(polys).bounds
    return max(bounds[2] - bounds[0], bounds[3] - bounds[1])

def get_side_from_polys(polys: List[Polygon]) -> float:
    """Get side from polygon list."""
    if not polys:
        return 0.0
    bounds = unary_union(polys).bounds
    return max(bounds[2] - bounds[0], bounds[3] - bounds[1])

def center_solution(placements: Solution) -> Solution:
    """Center around origin."""
    if not placements:
        return placements
    polys = [make_polygon(*p) for p in placements]
    bounds = unary_union(polys).bounds
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]

def count_overlaps(placements: Solution) -> int:
    """Count overlapping pairs."""
    polys = [make_polygon(*p) for p in placements]
    count = 0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                count += 1
    return count

def is_valid(placements: Solution) -> bool:
    """Check if solution is valid."""
    return count_overlaps(placements) == 0

def repair_solution(placements: Solution, max_iters: int = 1000) -> Solution:
    """Try to repair invalid solution."""
    sol = list(placements)
    n = len(sol)

    for _ in range(max_iters):
        polys = [make_polygon(*p) for p in sol]

        # Find overlapping pairs
        overlaps = []
        for i in range(n):
            for j in range(i + 1, n):
                if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                    overlaps.append((i, j))

        if not overlaps:
            return sol

        # Move one tree from random overlapping pair
        i, j = random.choice(overlaps)
        idx = random.choice([i, j])
        x, y, d = sol[idx]

        # Try random displacement
        for _ in range(20):
            nx = x + random.gauss(0, 0.1)
            ny = y + random.gauss(0, 0.1)
            nd = (d + random.gauss(0, 15)) % 360

            new_poly = make_polygon(nx, ny, nd)
            others = polys[:idx] + polys[idx+1:]

            if not check_collision(new_poly, others):
                sol[idx] = (nx, ny, nd)
                break

    return sol

# ============================================================================
# INITIAL PLACEMENT STRATEGIES
# ============================================================================

def fermat_spiral(n: int, scale: float = 0.45) -> List[Tuple[float, float]]:
    """Fermat spiral positions."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    return [(scale * math.sqrt(i) * math.cos(i * golden),
             scale * math.sqrt(i) * math.sin(i * golden)) for i in range(n)]

def sunflower_pattern(n: int, alpha: float = 1.0) -> List[Tuple[float, float]]:
    """Sunflower pattern (variant of Fermat spiral)."""
    positions = []
    golden = (1 + math.sqrt(5)) / 2

    for i in range(n):
        theta = 2 * math.pi * i / (golden ** 2)
        r = 0.45 * math.sqrt(i) * alpha
        positions.append((r * math.cos(theta), r * math.sin(theta)))

    return positions

def hexagonal_close_pack(n: int, spacing: float = 0.7) -> List[Tuple[float, float]]:
    """Hexagonal close packing."""
    positions = [(0.0, 0.0)]
    if n <= 1:
        return positions[:n]

    dx = spacing * 0.7
    dy = spacing * 0.7 * math.sqrt(3) / 2

    ring = 1
    while len(positions) < n:
        # Generate ring
        for side in range(6):
            for step in range(ring):
                if len(positions) >= n:
                    break
                angle = math.pi / 3 * side + math.pi / 6
                x = ring * dx * math.cos(angle) - step * dx * math.cos(angle + math.pi / 3)
                y = ring * dy * math.sin(angle) - step * dy * math.sin(angle + math.pi / 3)
                positions.append((x, y))
        ring += 1

    return positions[:n]

def place_radially(polys: List[Polygon], idx: Optional[STRtree], attempts: int = 100) -> Placement:
    """Place tree radially."""
    if not polys:
        return (0.0, 0.0, random.uniform(0, 360))

    best = None
    best_r = float('inf')

    for _ in range(attempts):
        # Weighted angle
        while True:
            theta = random.uniform(0, 2 * math.pi)
            if random.random() < abs(math.sin(2 * theta)) + 0.15:
                break

        vx, vy = math.cos(theta), math.sin(theta)
        lo, hi = 0.0, 15.0

        while hi - lo > 0.01:
            mid = (lo + hi) / 2
            found = False
            for rot in range(0, 360, 20):
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
            for rot in range(0, 360, 10):
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

def build_solution(n: int, strategy: str = 'spiral', scale: float = 0.45) -> Solution:
    """Build initial solution."""
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    if strategy == 'spiral':
        positions = fermat_spiral(n, scale)
    elif strategy == 'sunflower':
        positions = sunflower_pattern(n, scale)
    else:
        positions = hexagonal_close_pack(n, scale)

    placements = []
    polys = []

    for px, py in positions:
        if not polys:
            placements.append((px, py, 0.0))
            polys.append(make_polygon(px, py, 0.0))
        else:
            idx = STRtree(polys)
            found = None
            for rot in range(0, 360, 10):
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
# CMA-ES OPTIMIZER
# ============================================================================

class CMAES:
    """Covariance Matrix Adaptation Evolution Strategy."""

    def __init__(self, n_vars: int, sigma: float = 0.3, population_size: int = None):
        self.n = n_vars
        self.sigma = sigma
        self.pop_size = population_size or (4 + int(3 * math.log(n_vars)))

        # Strategy parameters
        self.mu = self.pop_size // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mueff = 1.0 / (self.weights ** 2).sum()

        # Adaptation parameters
        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs

        # Evolution paths
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)

        # Covariance matrix
        self.C = np.eye(self.n)
        self.B = np.eye(self.n)
        self.D = np.ones(self.n)

        self.chiN = math.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))

        self.mean = None
        self.generation = 0

    def ask(self, mean: np.ndarray) -> np.ndarray:
        """Generate population."""
        self.mean = mean.copy()

        # Sample population
        samples = np.zeros((self.pop_size, self.n))
        for i in range(self.pop_size):
            z = np.random.randn(self.n)
            samples[i] = mean + self.sigma * (self.B @ (self.D * z))

        return samples

    def tell(self, samples: np.ndarray, fitness: np.ndarray):
        """Update parameters based on fitness."""
        # Sort by fitness
        order = np.argsort(fitness)
        samples = samples[order]

        # Update mean
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, None] * samples[:self.mu], axis=0)

        # Update evolution paths
        y = (self.mean - old_mean) / self.sigma
        z = np.linalg.solve(self.B @ np.diag(self.D), y)

        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * z

        hsig = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) < (1.4 + 2 / (self.n + 1)) * self.chiN

        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y

        # Update covariance matrix
        artmp = (samples[:self.mu] - old_mean) / self.sigma
        self.C = (1 - self.c1 - self.cmu) * self.C + \
                 self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                 self.cmu * (artmp.T @ np.diag(self.weights) @ artmp)

        # Update sigma
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

        # Decompose C
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        D2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(D2, 1e-20))

        self.generation += 1

        return self.mean

def cmaes_optimize(placements: Solution, generations: int = 100,
                   population_size: int = 20, sigma: float = 0.3) -> Solution:
    """Optimize using CMA-ES."""
    n = len(placements)
    if n <= 1:
        return placements

    # Encode solution as flat array [x1, y1, d1, x2, y2, d2, ...]
    def encode(sol: Solution) -> np.ndarray:
        arr = []
        for x, y, d in sol:
            arr.extend([x, y, d / 360.0])  # Normalize angle
        return np.array(arr)

    def decode(arr: np.ndarray) -> Solution:
        sol = []
        for i in range(0, len(arr), 3):
            x, y, d = arr[i], arr[i+1], (arr[i+2] % 1.0) * 360.0
            sol.append((x, y, d))
        return sol

    def fitness(arr: np.ndarray) -> float:
        sol = decode(arr)
        if not is_valid(sol):
            # Penalize invalid solutions
            return 1000.0 + count_overlaps(sol) * 10.0
        return get_bounding_side(sol)

    # Initialize CMA-ES
    mean = encode(placements)
    cmaes = CMAES(len(mean), sigma=sigma, population_size=population_size)

    best_sol = placements
    best_score = get_bounding_side(placements) if is_valid(placements) else float('inf')

    for gen in range(generations):
        # Generate and evaluate population
        samples = cmaes.ask(mean)
        fit = np.array([fitness(s) for s in samples])

        # Update CMA-ES
        mean = cmaes.tell(samples, fit)

        # Track best
        best_idx = np.argmin(fit)
        if fit[best_idx] < best_score:
            candidate = decode(samples[best_idx])
            if is_valid(candidate):
                best_score = fit[best_idx]
                best_sol = candidate

    return best_sol

# ============================================================================
# DIFFERENTIAL EVOLUTION
# ============================================================================

def differential_evolution(placements: Solution, generations: int = 80,
                          population_size: int = 30, F: float = 0.8,
                          CR: float = 0.9) -> Solution:
    """Optimize using Differential Evolution."""
    n = len(placements)
    if n <= 1:
        return placements

    def encode(sol: Solution) -> np.ndarray:
        return np.array([[x, y, d] for x, y, d in sol]).flatten()

    def decode(arr: np.ndarray) -> Solution:
        arr = arr.reshape(-1, 3)
        return [(x, y, d % 360) for x, y, d in arr]

    def fitness(arr: np.ndarray) -> float:
        sol = decode(arr)
        if not is_valid(sol):
            return 1000.0 + count_overlaps(sol) * 10.0
        return get_bounding_side(sol)

    # Initialize population
    dim = n * 3
    base = encode(placements)
    population = [base + np.random.randn(dim) * 0.3 for _ in range(population_size)]
    population[0] = base  # Keep original

    fit = np.array([fitness(p) for p in population])

    best_idx = np.argmin(fit)
    best_sol = decode(population[best_idx]) if fit[best_idx] < 1000 else placements
    best_score = min(fit[best_idx], get_bounding_side(placements) if is_valid(placements) else float('inf'))

    for gen in range(generations):
        for i in range(population_size):
            # Select three random vectors
            candidates = list(range(population_size))
            candidates.remove(i)
            a, b, c = random.sample(candidates, 3)

            # Mutation
            mutant = population[a] + F * (population[b] - population[c])

            # Crossover
            trial = np.copy(population[i])
            j_rand = random.randint(0, dim - 1)
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]

            # Selection
            trial_fit = fitness(trial)
            if trial_fit < fit[i]:
                population[i] = trial
                fit[i] = trial_fit

                if trial_fit < best_score:
                    candidate = decode(trial)
                    if is_valid(candidate):
                        best_score = trial_fit
                        best_sol = candidate

    return best_sol

# ============================================================================
# PARALLEL TEMPERING
# ============================================================================

def parallel_tempering(placements: Solution, iterations: int = 50000,
                      temperatures: List[float] = None,
                      exchange_every: int = 100) -> Solution:
    """Parallel Tempering with replica exchange."""
    n = len(placements)
    if n <= 1:
        return placements

    if temperatures is None:
        temperatures = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    num_replicas = len(temperatures)

    # Initialize replicas
    replicas = [list(placements) for _ in range(num_replicas)]
    polys_list = [[make_polygon(*p) for p in r] for r in replicas]
    scores = [get_side_from_polys(p) for p in polys_list]

    best_sol = placements
    best_score = get_bounding_side(placements) if is_valid(placements) else float('inf')

    # Precompute shift sizes
    shift_sizes = [s * 0.08 for s in scores]

    for it in range(iterations):
        # Update each replica
        for r in range(num_replicas):
            T = temperatures[r]
            current = replicas[r]
            polys = polys_list[r]
            current_score = scores[r]

            # Random move
            i = random.randrange(n)
            x, y, d = current[i]

            shift = shift_sizes[r]
            if random.random() < 0.5:
                nx = x + random.gauss(0, shift)
                ny = y + random.gauss(0, shift)
                nd = d
            elif random.random() < 0.8:
                nx, ny = x, y
                nd = (d + random.gauss(0, 30)) % 360
            else:
                nx = x + random.gauss(0, shift * 0.6)
                ny = y + random.gauss(0, shift * 0.6)
                nd = (d + random.gauss(0, 20)) % 360

            new_poly = make_polygon(nx, ny, nd)
            others = polys[:i] + polys[i+1:]

            if check_collision(new_poly, others):
                continue

            old_poly = polys[i]
            polys[i] = new_poly
            new_score = get_side_from_polys(polys)

            delta = new_score - current_score

            if delta <= 0 or random.random() < math.exp(-delta / T):
                current[i] = (nx, ny, nd)
                scores[r] = new_score

                # Update best (only from lowest temperature)
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

                # Metropolis criterion for exchange
                delta = (1/T_lo - 1/T_hi) * (E_hi - E_lo)
                if delta < 0 or random.random() < math.exp(-delta):
                    # Swap replicas
                    replicas[r], replicas[r+1] = replicas[r+1], replicas[r]
                    polys_list[r], polys_list[r+1] = polys_list[r+1], polys_list[r]
                    scores[r], scores[r+1] = scores[r+1], scores[r]
                    shift_sizes[r], shift_sizes[r+1] = shift_sizes[r+1], shift_sizes[r]

        # Adaptive shift sizes
        if it % 1000 == 0:
            for r in range(num_replicas):
                shift_sizes[r] = scores[r] * 0.06

    return best_sol

# ============================================================================
# ULTRA-LONG SIMULATED ANNEALING
# ============================================================================

def ultra_sa(placements: Solution, iterations: int = 200000,
            temp_start: float = 5.0, temp_end: float = 1e-12,
            reheat_every: int = 20000, reheat_factor: float = 0.15) -> Solution:
    """Ultra-long simulated annealing with multiple reheats."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_polygon(*p) for p in current]
    current_score = get_side_from_polys(polys)

    best = list(current)
    best_score = current_score

    T = temp_start
    cooling = (temp_end / temp_start) ** (1.0 / iterations)

    shift = current_score * 0.1
    rot = 40.0

    stagnation = 0
    last_improve = 0

    for it in range(iterations):
        # Reheat
        if it > 0 and it % reheat_every == 0:
            T = max(T, temp_start * reheat_factor * (1 - it / iterations))
            shift = best_score * 0.08
            rot = 35.0

        i = random.randrange(n)
        x, y, d = current[i]

        r = random.random()
        if r < 0.45:
            nx = x + random.gauss(0, shift)
            ny = y + random.gauss(0, shift)
            nd = d
        elif r < 0.75:
            nx, ny = x, y
            nd = (d + random.gauss(0, rot)) % 360
        elif r < 0.92:
            nx = x + random.gauss(0, shift * 0.6)
            ny = y + random.gauss(0, shift * 0.6)
            nd = (d + random.gauss(0, rot * 0.6)) % 360
        else:
            nx = x + random.gauss(0, shift * 2)
            ny = y + random.gauss(0, shift * 2)
            nd = random.uniform(0, 360)

        new_poly = make_polygon(nx, ny, nd)
        others = polys[:i] + polys[i+1:]

        if check_collision(new_poly, others):
            T *= cooling
            stagnation += 1
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_score = get_side_from_polys(polys)

        delta = new_score - current_score

        if delta <= 0 or random.random() < math.exp(-delta / T):
            current[i] = (nx, ny, nd)
            current_score = new_score
            stagnation = 0

            if new_score < best_score:
                best_score = new_score
                best = list(current)
                last_improve = it
        else:
            polys[i] = old_poly

        T *= cooling

        # Adaptive
        if stagnation > 500:
            shift = max(0.002, shift * 0.95)
            rot = max(0.5, rot * 0.95)
            stagnation = 0

    return best

# ============================================================================
# COMPACTION
# ============================================================================

def aggressive_compact(placements: Solution, iterations: int = 50000) -> Solution:
    """Aggressive compaction toward center."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_polygon(*p) for p in current]
    current_score = get_side_from_polys(polys)

    for it in range(iterations):
        bounds = unary_union(polys).bounds
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2

        i = random.randrange(n)
        x, y, d = current[i]

        dx, dy = cx - x, cy - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.01:
            continue

        step = random.uniform(0.002, min(0.05, dist * 0.3))
        nx = x + step * dx / dist
        ny = y + step * dy / dist
        nd = d

        if random.random() < 0.15:
            nd = (d + random.gauss(0, 8)) % 360

        new_poly = make_polygon(nx, ny, nd)
        others = polys[:i] + polys[i+1:]

        if check_collision(new_poly, others):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_score = get_side_from_polys(polys)

        if new_score <= current_score:
            current[i] = (nx, ny, nd)
            current_score = new_score
        else:
            polys[i] = old_poly

    return current

def squeeze_boundary(placements: Solution, iterations: int = 20000) -> Solution:
    """Squeeze boundary trees inward."""
    n = len(placements)
    if n <= 2:
        return placements

    current = list(placements)
    polys = [make_polygon(*p) for p in current]

    for _ in range(iterations):
        bounds = unary_union(polys).bounds
        minx, miny, maxx, maxy = bounds
        side = max(maxx - minx, maxy - miny)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

        # Find boundary trees
        boundary = []
        for i in range(n):
            pb = polys[i].bounds
            on_b = (abs(pb[0] - minx) < 0.03 or abs(pb[2] - maxx) < 0.03 or
                   abs(pb[1] - miny) < 0.03 or abs(pb[3] - maxy) < 0.03)
            if on_b:
                boundary.append(i)

        if not boundary:
            break

        i = random.choice(boundary)
        x, y, d = current[i]

        dx, dy = cx - x, cy - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.01:
            continue

        step = random.uniform(0.003, 0.02)
        nx = x + step * dx / dist
        ny = y + step * dy / dist

        new_poly = make_polygon(nx, ny, d)
        others = polys[:i] + polys[i+1:]

        if check_collision(new_poly, others):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_side = get_side_from_polys(polys)

        if new_side < side:
            current[i] = (nx, ny, d)
        else:
            polys[i] = old_poly

    return current

def optimize_rotations(placements: Solution, iterations: int = 20000) -> Solution:
    """Optimize rotations."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_polygon(*p) for p in current]
    current_score = get_side_from_polys(polys)

    for _ in range(iterations):
        i = random.randrange(n)
        x, y, d = current[i]
        nd = (d + random.gauss(0, 20)) % 360

        new_poly = make_polygon(x, y, nd)
        others = polys[:i] + polys[i+1:]

        if check_collision(new_poly, others):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_score = get_side_from_polys(polys)

        if new_score < current_score:
            current[i] = (x, y, nd)
            current_score = new_score
        else:
            polys[i] = old_poly

    return current

# ============================================================================
# MAIN SOLVER
# ============================================================================

class AdvancedPackingSolver:
    """Main solver combining all techniques."""

    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        random.seed(seed)
        np.random.seed(seed)

        self.solutions: Dict[int, Solution] = {}
        self.scores: Dict[int, float] = {}

        # Cache for optimal small n solutions
        self.optimal_cache: Dict[int, Solution] = {}

    def optimize_single(self, n: int) -> Solution:
        """Optimize for a single n using all techniques."""
        if n == 0:
            return []
        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            self.solutions[1] = sol
            self.scores[1] = get_bounding_side(sol)
            return sol

        best_sol = None
        best_score = float('inf')

        # Try multiple restarts with different strategies
        strategies = ['spiral', 'sunflower', 'hex']
        scales = [0.42, 0.45, 0.48, 0.50]

        for restart in range(CFG.num_restarts):
            strategy = strategies[restart % len(strategies)]
            scale = scales[restart % len(scales)]

            # Build initial solution
            if n - 1 in self.solutions and restart == 0:
                prev = self.solutions[n - 1]
                prev_polys = [make_polygon(*p) for p in prev]
                idx = STRtree(prev_polys)
                new_tree = place_radially(prev_polys, idx, attempts=100)
                sol = prev + [new_tree]
            else:
                random.seed(self.seed + n * 1000 + restart * 100)
                np.random.seed(self.seed + n * 1000 + restart * 100)
                sol = build_solution(n, strategy, scale)

            if not is_valid(sol):
                sol = repair_solution(sol)
                if not is_valid(sol):
                    continue

            # Scale iterations with n
            sa_iters = CFG.sa_iterations + n * 500
            cmaes_gens = max(30, CFG.cmaes_generations - n // 5)
            de_gens = max(20, CFG.de_generations - n // 4)
            pt_iters = max(20000, CFG.pt_iterations - n * 100)

            # Phase 1: CMA-ES (for smaller n)
            if n <= 100:
                sol = cmaes_optimize(sol, generations=cmaes_gens,
                                    population_size=CFG.cmaes_population)

            # Phase 2: Differential Evolution
            if n <= 80:
                sol = differential_evolution(sol, generations=de_gens,
                                           population_size=CFG.de_population)

            # Phase 3: Parallel Tempering
            sol = parallel_tempering(sol, iterations=pt_iters,
                                    temperatures=CFG.pt_temperatures)

            # Phase 4: Ultra-long SA
            sol = ultra_sa(sol, iterations=sa_iters)

            # Phase 5: Compaction
            sol = aggressive_compact(sol, CFG.compact_iterations)
            sol = squeeze_boundary(sol, CFG.compact_iterations // 2)

            # Phase 6: Rotation optimization
            sol = optimize_rotations(sol, CFG.compact_iterations // 2)

            # Phase 7: Final SA
            sol = ultra_sa(sol, iterations=sa_iters // 2,
                          temp_start=1.0, temp_end=1e-12)

            # Phase 8: Final compaction
            sol = aggressive_compact(sol, CFG.compact_iterations // 2)

            # Validate
            if not is_valid(sol):
                sol = repair_solution(sol)
                if not is_valid(sol):
                    continue

            score = get_bounding_side(sol)
            if score < best_score:
                best_score = score
                best_sol = sol

        # Fallback
        if best_sol is None:
            if n - 1 in self.solutions:
                prev = self.solutions[n - 1]
                prev_polys = [make_polygon(*p) for p in prev]
                idx = STRtree(prev_polys)
                new_tree = place_radially(prev_polys, idx)
                best_sol = prev + [new_tree]
            else:
                best_sol = build_solution(n, 'spiral', 0.55)

        best_sol = center_solution(best_sol)
        self.solutions[n] = best_sol
        self.scores[n] = get_bounding_side(best_sol)

        return best_sol

    def solve_all(self, max_n: int = 200) -> Dict[int, Solution]:
        """Solve for all n."""
        start = time.time()

        if self.verbose:
            print(f"Advanced Packing Solver: n=1 to {max_n}, seed={self.seed}")
            print(f"Techniques: CMA-ES, DE, Parallel Tempering, Ultra-SA")
            print("=" * 70)

        for n in range(1, max_n + 1):
            self.optimize_single(n)

            if self.verbose and (n % 5 == 0 or n <= 5):
                elapsed = time.time() - start
                score = self.total_score()
                eta = (elapsed / n) * (max_n - n) if n > 0 else 0
                print(f"n={n:3d}: side={self.scores[n]:.4f}, score={score:.2f}, "
                      f"time={elapsed:.0f}s, ETA={eta/3600:.1f}h")

        if self.verbose:
            print("=" * 70)
            print(f"FINAL SCORE: {self.total_score():.4f}")
            print(f"Total time: {(time.time() - start)/3600:.2f}h")

        return self.solutions

    def total_score(self) -> float:
        """Compute competition score."""
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, get_bounding_side(sol))
            total += (side ** 2) / n
        return total

    def save(self, path: str = "solver_state.pkl"):
        """Save solver state."""
        with open(path, 'wb') as f:
            pickle.dump({
                'solutions': self.solutions,
                'scores': self.scores,
                'seed': self.seed
            }, f)

    def load(self, path: str = "solver_state.pkl"):
        """Load solver state."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.solutions = data['solutions']
                self.scores = data['scores']

# ============================================================================
# SUBMISSION
# ============================================================================

def create_submission(solutions: Dict[int, Solution], path: str = "submission.csv"):
    """Create submission file."""
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

def validate_all(solutions: Dict[int, Solution]) -> Tuple[bool, List[str]]:
    """Validate all solutions."""
    errors = []
    for n in range(1, 201):
        if n not in solutions:
            errors.append(f"Missing n={n}")
        elif len(solutions[n]) != n:
            errors.append(f"n={n}: wrong count")
        elif not is_valid(solutions[n]):
            errors.append(f"n={n}: overlaps")
    return len(errors) == 0, errors

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ADVANCED TREE PACKING SOLVER")
    print("CMA-ES + Differential Evolution + Parallel Tempering + Ultra-SA")
    print("Target: Score < 60")
    print("=" * 70)
    print()

    best_score = float('inf')
    best_solutions = None

    for seed in CFG.seeds:
        print(f"\n{'='*70}")
        print(f"Seed: {seed}")
        print(f"{'='*70}")

        solver = AdvancedPackingSolver(seed=seed, verbose=True)
        solutions = solver.solve_all(max_n=200)

        # Save intermediate
        solver.save(f"state_seed_{seed}.pkl")

        valid, errors = validate_all(solutions)
        if not valid:
            print(f"Validation errors: {errors[:3]}")
            continue

        score = solver.total_score()
        print(f"Seed {seed} score: {score:.4f}")

        if score < best_score:
            best_score = score
            best_solutions = copy.deepcopy(solutions)
            print(f"*** NEW BEST: {score:.4f} ***")

            # Save best submission
            create_submission(best_solutions, "submission.csv")

    if best_solutions:
        print(f"\n{'='*70}")
        print(f"BEST SCORE: {best_score:.4f}")
        create_submission(best_solutions, "submission.csv")
        print("Saved: submission.csv")

        valid, _ = validate_all(best_solutions)
        print(f"Valid: {valid}")

if __name__ == "__main__":
    main()
