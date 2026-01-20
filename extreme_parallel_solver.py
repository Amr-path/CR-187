#!/usr/bin/env python3
"""
EXTREME PARALLEL TREE PACKING SOLVER - MAXIMUM QUALITY
======================================================

Optimized for VPS with 14 cores. Runs in EXTREME mode only.
No lite options - full power optimization.

Features:
- 14 cores parallel processing by default
- Ultra-long SA iterations (1M+)
- Extended Parallel Tempering (16 replicas)
- CMA-ES + Differential Evolution
- Exhaustive search for small n
- Aggressive multi-phase compaction
- Auto checkpointing and resume

Target: Score < 55

Run: python extreme_parallel_solver.py
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
from dataclasses import dataclass, field
import multiprocessing as mp
from functools import partial
import numpy as np

from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.ops import unary_union

# ============================================================================
# EXTREME CONFIGURATION - NO LITE OPTIONS
# ============================================================================

@dataclass
class ExtremeConfig:
    """Maximum quality configuration."""

    # Parallel processing - 14 cores default for VPS
    num_cores: int = 14
    num_seeds: int = 14  # One seed per core

    # Seeds for parallel runs
    seeds: List[int] = field(default_factory=lambda: [
        42, 123, 456, 789, 2025, 1337, 7777, 9999,
        31415, 27182, 161803, 14142, 17320, 22360
    ])

    # Extreme SA parameters
    sa_base_iterations: int = 800000
    sa_scale_per_n: int = 2000
    sa_temp_start: float = 10.0
    sa_temp_end: float = 1e-15
    sa_reheat_every: int = 40000
    sa_reheat_factor: float = 0.12

    # Extended Parallel Tempering
    pt_base_iterations: int = 150000
    pt_num_replicas: int = 16
    pt_exchange_every: int = 40
    pt_temp_min: float = 0.0005
    pt_temp_max: float = 25.0

    # CMA-ES (extended for extreme mode)
    cmaes_population: int = 30
    cmaes_base_generations: int = 150
    cmaes_sigma: float = 0.25
    cmaes_max_n: int = 120  # Apply to larger n

    # Differential Evolution (extended)
    de_population: int = 40
    de_base_generations: int = 120
    de_F: float = 0.75
    de_CR: float = 0.92
    de_max_n: int = 100

    # Compaction (extreme)
    compact_base_iterations: int = 80000
    squeeze_base_iterations: int = 40000
    rotation_base_iterations: int = 35000

    # Multi-restart
    num_restarts: int = 8

    # Exhaustive search for small n
    exhaustive_max_n: int = 12
    exhaustive_angle_step: int = 3
    exhaustive_max_iters: int = 2000000

    # Radial placement
    radial_attempts: int = 200

    # Checkpointing
    checkpoint_every: int = 3

CFG = ExtremeConfig()

# ============================================================================
# GEOMETRY (Optimized)
# ============================================================================

TREE_COORDS = np.array([
    [0.0, 0.8], [0.125, 0.5], [0.0625, 0.5], [0.2, 0.25], [0.1, 0.25],
    [0.35, 0.0], [0.075, 0.0], [0.075, -0.2], [-0.075, -0.2], [-0.075, 0.0],
    [-0.35, 0.0], [-0.1, 0.25], [-0.2, 0.25], [-0.0625, 0.5], [-0.125, 0.5],
], dtype=np.float64)

BASE_POLYGON = Polygon(TREE_COORDS)

Placement = Tuple[float, float, float]
Solution = List[Placement]

# Pre-compute rotation matrices for common angles
_ROTATION_CACHE = {}

def _get_rotation_matrix(angle: float) -> np.ndarray:
    """Get cached rotation matrix."""
    key = round(angle * 100)
    if key not in _ROTATION_CACHE:
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        _ROTATION_CACHE[key] = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=np.float64)
    return _ROTATION_CACHE[key]

def make_polygon(x: float, y: float, angle: float) -> Polygon:
    """Create tree polygon (optimized)."""
    rot_mat = _get_rotation_matrix(angle)
    rotated = TREE_COORDS @ rot_mat
    translated = rotated + np.array([x, y])
    return Polygon(translated)

def check_collision(poly: Polygon, others: List[Polygon]) -> bool:
    """Check collision (touching OK)."""
    for o in others:
        if poly.intersects(o) and not poly.touches(o):
            return True
    return False

def check_collision_indexed(poly: Polygon, idx: STRtree, polys: List[Polygon]) -> bool:
    """Fast indexed collision check."""
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
    """Get side from polygon list (faster)."""
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
    """Check if solution is valid (no overlaps)."""
    return count_overlaps(placements) == 0

def repair_solution(placements: Solution, max_iters: int = 2000) -> Solution:
    """Repair invalid solution."""
    sol = list(placements)
    n = len(sol)

    for _ in range(max_iters):
        polys = [make_polygon(*p) for p in sol]
        overlaps = []
        for i in range(n):
            for j in range(i + 1, n):
                if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                    overlaps.append((i, j))

        if not overlaps:
            return sol

        i, j = random.choice(overlaps)
        idx = random.choice([i, j])
        x, y, d = sol[idx]

        for _ in range(30):
            nx = x + random.gauss(0, 0.15)
            ny = y + random.gauss(0, 0.15)
            nd = (d + random.gauss(0, 20)) % 360

            new_poly = make_polygon(nx, ny, nd)
            others = polys[:idx] + polys[idx+1:]

            if not check_collision(new_poly, others):
                sol[idx] = (nx, ny, nd)
                break

    return sol

# ============================================================================
# PLACEMENT STRATEGIES
# ============================================================================

def fermat_spiral(n: int, scale: float = 0.42) -> List[Tuple[float, float]]:
    """Fermat spiral positions."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    return [(scale * math.sqrt(i) * math.cos(i * golden),
             scale * math.sqrt(i) * math.sin(i * golden)) for i in range(n)]

def sunflower_pattern(n: int, scale: float = 0.45) -> List[Tuple[float, float]]:
    """Sunflower pattern."""
    positions = []
    golden = (1 + math.sqrt(5)) / 2
    for i in range(n):
        theta = 2 * math.pi * i / (golden ** 2)
        r = scale * math.sqrt(i)
        positions.append((r * math.cos(theta), r * math.sin(theta)))
    return positions

def hexagonal_close_pack(n: int, spacing: float = 0.65) -> List[Tuple[float, float]]:
    """Hexagonal close packing."""
    positions = [(0.0, 0.0)]
    if n <= 1:
        return positions[:n]

    dx = spacing * 0.7
    dy = spacing * 0.7 * math.sqrt(3) / 2

    ring = 1
    while len(positions) < n:
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

def place_radially(polys: List[Polygon], idx: Optional[STRtree],
                   attempts: int = None) -> Placement:
    """Place tree radially with extreme precision."""
    attempts = attempts or CFG.radial_attempts

    if not polys:
        return (0.0, 0.0, random.uniform(0, 360))

    best = None
    best_r = float('inf')

    for _ in range(attempts):
        while True:
            theta = random.uniform(0, 2 * math.pi)
            if random.random() < abs(math.sin(2 * theta)) + 0.12:
                break

        vx, vy = math.cos(theta), math.sin(theta)
        lo, hi = 0.0, 15.0

        # Finer binary search
        while hi - lo > 0.003:
            mid = (lo + hi) / 2
            found = False
            for rot in range(0, 360, 8):
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
            for rot in range(0, 360, 3):  # Finer rotation search
                p = make_polygon(px, py, rot)
                if idx:
                    coll = check_collision_indexed(p, idx, polys)
                else:
                    coll = check_collision(p, polys)
                if not coll:
                    best_r = hi
                    best = (px, py, rot)
                    break

    return best if best else (12.0, 0.0, 0.0)

def build_solution(n: int, strategy: str = 'spiral', scale: float = 0.42) -> Solution:
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
# EXHAUSTIVE OPTIMIZATION (Small n)
# ============================================================================

def exhaustive_optimize(n: int, base_solution: Solution) -> Solution:
    """Exhaustive optimization for small n (extreme mode)."""
    if n <= 1 or n > CFG.exhaustive_max_n:
        return base_solution

    best_sol = list(base_solution)
    best_score = get_bounding_side(best_sol)

    current = list(base_solution)
    iters = 0
    max_iters = CFG.exhaustive_max_iters // max(1, n - 2)

    for _ in range(max_iters // (n * 120)):
        improved = False

        for i in range(n):
            x, y, d = current[i]

            for new_d in range(0, 360, CFG.exhaustive_angle_step):
                for dx in np.linspace(-0.12, 0.12, 7):
                    for dy in np.linspace(-0.12, 0.12, 7):
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
# CMA-ES OPTIMIZER (Extended)
# ============================================================================

class CMAES:
    """CMA-ES implementation for extreme optimization."""

    def __init__(self, n_vars: int, sigma: float = 0.25, population_size: int = None):
        self.n = n_vars
        self.sigma = sigma
        self.pop_size = population_size or max(8, 4 + int(3 * math.log(n_vars)))

        self.mu = self.pop_size // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mueff = 1.0 / (self.weights ** 2).sum()

        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs

        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.C = np.eye(self.n)
        self.B = np.eye(self.n)
        self.D = np.ones(self.n)

        self.chiN = math.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))
        self.mean = None
        self.generation = 0

    def ask(self, mean: np.ndarray) -> np.ndarray:
        self.mean = mean.copy()
        samples = np.zeros((self.pop_size, self.n))
        for i in range(self.pop_size):
            z = np.random.randn(self.n)
            samples[i] = mean + self.sigma * (self.B @ (self.D * z))
        return samples

    def tell(self, samples: np.ndarray, fitness: np.ndarray):
        order = np.argsort(fitness)
        samples = samples[order]

        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, None] * samples[:self.mu], axis=0)

        y = (self.mean - old_mean) / self.sigma
        z = np.linalg.solve(self.B @ np.diag(self.D), y)

        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * z

        hsig = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) < (1.4 + 2 / (self.n + 1)) * self.chiN

        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y

        artmp = (samples[:self.mu] - old_mean) / self.sigma
        self.C = (1 - self.c1 - self.cmu) * self.C + \
                 self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                 self.cmu * (artmp.T @ np.diag(self.weights) @ artmp)

        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        D2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(D2, 1e-20))

        self.generation += 1
        return self.mean

def cmaes_optimize(placements: Solution, generations: int = None,
                   population_size: int = None) -> Solution:
    """Optimize using CMA-ES (extreme mode)."""
    n = len(placements)
    if n <= 1 or n > CFG.cmaes_max_n:
        return placements

    generations = generations or max(50, CFG.cmaes_base_generations - n)
    population_size = population_size or CFG.cmaes_population

    def encode(sol: Solution) -> np.ndarray:
        arr = []
        for x, y, d in sol:
            arr.extend([x, y, d / 360.0])
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
            return 1000.0 + count_overlaps(sol) * 10.0
        return get_bounding_side(sol)

    mean = encode(placements)
    cmaes = CMAES(len(mean), sigma=CFG.cmaes_sigma, population_size=population_size)

    best_sol = placements
    best_score = get_bounding_side(placements) if is_valid(placements) else float('inf')

    for gen in range(generations):
        samples = cmaes.ask(mean)
        fit = np.array([fitness(s) for s in samples])
        mean = cmaes.tell(samples, fit)

        best_idx = np.argmin(fit)
        if fit[best_idx] < best_score:
            candidate = decode(samples[best_idx])
            if is_valid(candidate):
                best_score = fit[best_idx]
                best_sol = candidate

    return best_sol

# ============================================================================
# DIFFERENTIAL EVOLUTION (Extended)
# ============================================================================

def differential_evolution(placements: Solution, generations: int = None,
                          population_size: int = None) -> Solution:
    """Differential Evolution (extreme mode)."""
    n = len(placements)
    if n <= 1 or n > CFG.de_max_n:
        return placements

    generations = generations or max(40, CFG.de_base_generations - n)
    population_size = population_size or CFG.de_population

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

    dim = n * 3
    base = encode(placements)
    population = [base + np.random.randn(dim) * 0.25 for _ in range(population_size)]
    population[0] = base

    fit = np.array([fitness(p) for p in population])

    best_idx = np.argmin(fit)
    best_sol = decode(population[best_idx]) if fit[best_idx] < 1000 else placements
    best_score = min(fit[best_idx], get_bounding_side(placements) if is_valid(placements) else float('inf'))

    for gen in range(generations):
        for i in range(population_size):
            candidates = list(range(population_size))
            candidates.remove(i)
            a, b, c = random.sample(candidates, 3)

            mutant = population[a] + CFG.de_F * (population[b] - population[c])

            trial = np.copy(population[i])
            j_rand = random.randint(0, dim - 1)
            for j in range(dim):
                if random.random() < CFG.de_CR or j == j_rand:
                    trial[j] = mutant[j]

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
# EXTENDED PARALLEL TEMPERING (16 replicas)
# ============================================================================

def extended_parallel_tempering(placements: Solution, iterations: int = None) -> Solution:
    """Extended Parallel Tempering with 16 replicas."""
    n = len(placements)
    if n <= 1:
        return placements

    iterations = iterations or max(50000, CFG.pt_base_iterations - n * 300)
    num_replicas = CFG.pt_num_replicas

    # Geometric temperature ladder
    temperatures = [CFG.pt_temp_min * (CFG.pt_temp_max / CFG.pt_temp_min) ** (i / (num_replicas - 1))
                   for i in range(num_replicas)]

    replicas = [list(placements) for _ in range(num_replicas)]
    polys_list = [[make_polygon(*p) for p in r] for r in replicas]
    scores = [get_side_from_polys(p) for p in polys_list]

    best_sol = placements
    best_score = get_bounding_side(placements) if is_valid(placements) else float('inf')

    shift_sizes = [s * 0.07 for s in scores]

    for it in range(iterations):
        for r in range(num_replicas):
            T = temperatures[r]
            current = replicas[r]
            polys = polys_list[r]
            current_score = scores[r]

            i = random.randrange(n)
            x, y, d = current[i]

            shift = shift_sizes[r]
            rnd = random.random()
            if rnd < 0.45:
                nx = x + random.gauss(0, shift)
                ny = y + random.gauss(0, shift)
                nd = d
            elif rnd < 0.75:
                nx, ny = x, y
                nd = (d + random.gauss(0, 30)) % 360
            elif rnd < 0.92:
                nx = x + random.gauss(0, shift * 0.5)
                ny = y + random.gauss(0, shift * 0.5)
                nd = (d + random.gauss(0, 20)) % 360
            else:
                nx = x + random.gauss(0, shift * 2.5)
                ny = y + random.gauss(0, shift * 2.5)
                nd = random.uniform(0, 360)

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

                if r == 0 and new_score < best_score:
                    best_score = new_score
                    best_sol = list(current)
            else:
                polys[i] = old_poly

        # Replica exchange
        if it % CFG.pt_exchange_every == 0 and it > 0:
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

        # Adaptive shift
        if it % 2000 == 0:
            for r in range(num_replicas):
                shift_sizes[r] = scores[r] * 0.055

    return best_sol

# ============================================================================
# EXTREME SIMULATED ANNEALING
# ============================================================================

def extreme_sa(placements: Solution, iterations: int = None,
              temp_start: float = None, temp_end: float = None) -> Solution:
    """Extreme SA with 800K+ iterations."""
    n = len(placements)
    if n <= 1:
        return placements

    iterations = iterations or max(400000, CFG.sa_base_iterations - n * CFG.sa_scale_per_n)
    temp_start = temp_start or CFG.sa_temp_start
    temp_end = temp_end or CFG.sa_temp_end

    current = list(placements)
    polys = [make_polygon(*p) for p in current]
    current_score = get_side_from_polys(polys)

    best = list(current)
    best_score = current_score

    T = temp_start
    cooling = (temp_end / temp_start) ** (1.0 / iterations)

    shift = current_score * 0.1
    rot = 45.0

    stagnation = 0

    for it in range(iterations):
        # Reheat
        if it > 0 and it % CFG.sa_reheat_every == 0:
            T = max(T, temp_start * CFG.sa_reheat_factor * (1 - it / iterations))
            shift = best_score * 0.08
            rot = 40.0

        i = random.randrange(n)
        x, y, d = current[i]

        r = random.random()
        if r < 0.42:
            nx = x + random.gauss(0, shift)
            ny = y + random.gauss(0, shift)
            nd = d
        elif r < 0.72:
            nx, ny = x, y
            nd = (d + random.gauss(0, rot)) % 360
        elif r < 0.90:
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
        else:
            polys[i] = old_poly

        T *= cooling

        # Adaptive
        if stagnation > 600:
            shift = max(0.002, shift * 0.94)
            rot = max(0.5, rot * 0.94)
            stagnation = 0

    return best

# ============================================================================
# MULTI-PHASE COMPACTION
# ============================================================================

def aggressive_compact(placements: Solution, iterations: int = None) -> Solution:
    """Aggressive center compaction."""
    n = len(placements)
    if n <= 1:
        return placements

    iterations = iterations or CFG.compact_base_iterations

    current = list(placements)
    polys = [make_polygon(*p) for p in current]
    current_score = get_side_from_polys(polys)

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

        step = random.uniform(0.001, min(0.05, dist * 0.3))
        nx = x + step * dx / dist
        ny = y + step * dy / dist
        nd = d

        if random.random() < 0.18:
            nd = (d + random.gauss(0, 6)) % 360

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

def squeeze_boundary(placements: Solution, iterations: int = None) -> Solution:
    """Squeeze boundary trees inward."""
    n = len(placements)
    if n <= 2:
        return placements

    iterations = iterations or CFG.squeeze_base_iterations

    current = list(placements)
    polys = [make_polygon(*p) for p in current]

    for _ in range(iterations):
        bounds = unary_union(polys).bounds
        minx, miny, maxx, maxy = bounds
        side = max(maxx - minx, maxy - miny)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

        boundary = []
        for i in range(n):
            pb = polys[i].bounds
            on_b = (abs(pb[0] - minx) < 0.025 or abs(pb[2] - maxx) < 0.025 or
                   abs(pb[1] - miny) < 0.025 or abs(pb[3] - maxy) < 0.025)
            if on_b:
                boundary.append(i)

        if not boundary:
            break

        i = random.choice(boundary)
        x, y, d = current[i]

        dx, dy = cx - x, cy - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.005:
            continue

        step = random.uniform(0.002, 0.025)
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

def optimize_rotations(placements: Solution, iterations: int = None) -> Solution:
    """Optimize rotations for tighter packing."""
    n = len(placements)
    if n <= 1:
        return placements

    iterations = iterations or CFG.rotation_base_iterations

    current = list(placements)
    polys = [make_polygon(*p) for p in current]
    current_score = get_side_from_polys(polys)

    for _ in range(iterations):
        i = random.randrange(n)
        x, y, d = current[i]
        nd = (d + random.gauss(0, 18)) % 360

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
# EXTREME SOLVER
# ============================================================================

class ExtremeParallelSolver:
    """Extreme quality solver with 14-core parallel processing."""

    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        random.seed(seed)
        np.random.seed(seed)

        self.solutions: Dict[int, Solution] = {}
        self.scores: Dict[int, float] = {}

        self.checkpoint_file = f"extreme_checkpoint_seed_{seed}.pkl"
        self.load_checkpoint()

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        print(f"\n[Seed {self.seed}] Saving checkpoint...")
        self.save_checkpoint()
        sys.exit(0)

    def save_checkpoint(self):
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump({
                'solutions': self.solutions,
                'scores': self.scores,
                'seed': self.seed
            }, f)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)
                self.solutions = data['solutions']
                self.scores = data['scores']
            if self.verbose:
                max_n = max(self.solutions.keys()) if self.solutions else 0
                print(f"[Seed {self.seed}] Resumed from n={max_n}")

    def optimize_single(self, n: int) -> Solution:
        """Optimize for single n with all extreme techniques."""
        if n == 0:
            return []
        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            self.solutions[1] = sol
            self.scores[1] = 1.0
            return sol

        best_sol = None
        best_score = float('inf')

        strategies = ['spiral', 'sunflower', 'hex']
        scales = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54]

        for restart in range(CFG.num_restarts):
            strategy = strategies[restart % len(strategies)]
            scale = scales[restart % len(scales)]

            # Build initial solution
            if n - 1 in self.solutions and restart == 0:
                prev = self.solutions[n - 1]
                prev_polys = [make_polygon(*p) for p in prev]
                idx = STRtree(prev_polys)
                new_tree = place_radially(prev_polys, idx)
                sol = prev + [new_tree]
            else:
                random.seed(self.seed + n * 1000 + restart * 100)
                np.random.seed(self.seed + n * 1000 + restart * 100)
                sol = build_solution(n, strategy, scale)

            if not is_valid(sol):
                sol = repair_solution(sol)
                if not is_valid(sol):
                    continue

            # Scale iterations based on n
            sa_iters = max(400000, CFG.sa_base_iterations - n * CFG.sa_scale_per_n)
            pt_iters = max(50000, CFG.pt_base_iterations - n * 300)
            compact_iters = max(40000, CFG.compact_base_iterations - n * 200)

            # Phase 1: Exhaustive for small n
            if n <= CFG.exhaustive_max_n:
                sol = exhaustive_optimize(n, sol)

            # Phase 2: CMA-ES
            if n <= CFG.cmaes_max_n:
                sol = cmaes_optimize(sol)

            # Phase 3: Differential Evolution
            if n <= CFG.de_max_n:
                sol = differential_evolution(sol)

            # Phase 4: Extended Parallel Tempering
            sol = extended_parallel_tempering(sol, iterations=pt_iters)

            # Phase 5: Extreme SA
            sol = extreme_sa(sol, iterations=sa_iters)

            # Phase 6: Multi-phase compaction
            sol = aggressive_compact(sol, compact_iters)
            sol = squeeze_boundary(sol, compact_iters // 2)
            sol = optimize_rotations(sol, compact_iters // 2)

            # Phase 7: Second SA pass
            sol = extreme_sa(sol, iterations=sa_iters // 2, temp_start=2.0)

            # Phase 8: Final compaction
            sol = aggressive_compact(sol, compact_iters // 2)
            sol = squeeze_boundary(sol, compact_iters // 3)

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
        start_n = max(self.solutions.keys()) + 1 if self.solutions else 1

        if self.verbose:
            print(f"[Seed {self.seed}] Extreme Solver: n={start_n} to {max_n}")

        for n in range(start_n, max_n + 1):
            iter_start = time.time()
            self.optimize_single(n)
            iter_time = time.time() - iter_start

            if self.verbose and (n % 5 == 0 or n <= 5):
                elapsed = time.time() - start
                score = self.total_score()
                eta = iter_time * (max_n - n)
                print(f"[Seed {self.seed}] n={n:3d}: side={self.scores[n]:.4f}, "
                      f"score={score:.2f}, time={iter_time:.0f}s, ETA={eta/3600:.1f}h")

            if n % CFG.checkpoint_every == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        return self.solutions

    def total_score(self) -> float:
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, get_bounding_side(sol))
            total += (side ** 2) / n
        return total

# ============================================================================
# PARALLEL EXECUTION
# ============================================================================

def solve_single_seed(seed: int, max_n: int = 200, verbose: bool = True) -> Tuple[float, Dict[int, Solution]]:
    """Solve with a single seed (for parallel execution)."""
    solver = ExtremeParallelSolver(seed=seed, verbose=verbose)
    solutions = solver.solve_all(max_n=max_n)
    return solver.total_score(), solutions

def run_parallel(seeds: List[int] = None, max_n: int = 200,
                 num_cores: int = None) -> Tuple[float, Dict[int, Solution]]:
    """Run multiple seeds in parallel on 14 cores."""
    seeds = seeds or CFG.seeds[:CFG.num_seeds]
    num_cores = num_cores or CFG.num_cores

    print("=" * 70)
    print("EXTREME PARALLEL TREE PACKING SOLVER")
    print(f"Cores: {num_cores} | Seeds: {len(seeds)} | Max n: {max_n}")
    print("=" * 70)

    start = time.time()

    with mp.Pool(num_cores) as pool:
        func = partial(solve_single_seed, max_n=max_n, verbose=True)
        results = pool.map(func, seeds)

    # Find best
    best_score = float('inf')
    best_solutions = None

    for score, solutions in results:
        if score < best_score:
            best_score = score
            best_solutions = solutions

    elapsed = time.time() - start
    print("=" * 70)
    print(f"BEST SCORE: {best_score:.4f}")
    print(f"Total time: {elapsed/3600:.2f} hours")

    return best_score, best_solutions

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
    print(f"Saved: {path}")
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
    """Main entry point - runs extreme parallel solver on 14 cores."""
    print("=" * 70)
    print("EXTREME PARALLEL TREE PACKING SOLVER")
    print("14 cores | Full Extreme Mode | No Lite Options")
    print("Target: Score < 55")
    print("=" * 70)
    print()

    best_score, best_solutions = run_parallel()

    if best_solutions:
        valid, errors = validate_all(best_solutions)
        print(f"Validation: {'PASSED' if valid else 'FAILED'}")

        if errors:
            print(f"Errors: {errors[:5]}")

        if valid:
            create_submission(best_solutions, "submission.csv")
            print(f"\nFINAL SCORE: {best_score:.4f}")

if __name__ == "__main__":
    main()
