#!/usr/bin/env python3
"""
Parallel Tree Packing Solver

Uses multiple CPU cores for parallel optimization.
Each seed runs in a separate process.

Run: python parallel_solver.py --cores 4
"""

import argparse
import math
import random
import time
import copy
import pickle
import os
from typing import List, Tuple, Dict
import multiprocessing as mp
from functools import partial
import numpy as np

from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# Import from main solver
from solver import (
    Config, CFG, make_polygon, check_collision, check_collision_indexed,
    get_bounding_side, get_side_from_polys, center_solution, count_overlaps,
    is_valid, repair_solution, build_solution, place_radially,
    cmaes_optimize, differential_evolution, parallel_tempering,
    ultra_sa, aggressive_compact, squeeze_boundary, optimize_rotations,
    create_submission, validate_all
)

Placement = Tuple[float, float, float]
Solution = List[Placement]

def solve_single_seed(seed: int, max_n: int = 200, verbose: bool = False) -> Tuple[float, Dict[int, Solution]]:
    """Solve with a single seed."""
    random.seed(seed)
    np.random.seed(seed)

    solutions: Dict[int, Solution] = {}
    scores: Dict[int, float] = {}

    start = time.time()

    for n in range(1, max_n + 1):
        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            solutions[1] = sol
            scores[1] = 1.0
            continue

        best_sol = None
        best_score = float('inf')

        strategies = ['spiral', 'sunflower', 'hex']
        scales = [0.42, 0.45, 0.48]

        for restart in range(3):
            strategy = strategies[restart % len(strategies)]
            scale = scales[restart % len(scales)]

            if n - 1 in solutions and restart == 0:
                prev = solutions[n - 1]
                prev_polys = [make_polygon(*p) for p in prev]
                idx = STRtree(prev_polys)
                new_tree = place_radially(prev_polys, idx, attempts=80)
                sol = prev + [new_tree]
            else:
                random.seed(seed + n * 1000 + restart * 100)
                np.random.seed(seed + n * 1000 + restart * 100)
                sol = build_solution(n, strategy, scale)

            if not is_valid(sol):
                sol = repair_solution(sol)
                if not is_valid(sol):
                    continue

            # Optimization
            sa_iters = 150000 + n * 400
            pt_iters = max(30000, 50000 - n * 100)

            # CMA-ES for small n
            if n <= 80:
                sol = cmaes_optimize(sol, generations=max(20, 60 - n//2), population_size=15)

            # DE for small n
            if n <= 60:
                sol = differential_evolution(sol, generations=max(15, 50 - n//2), population_size=20)

            # Parallel Tempering
            sol = parallel_tempering(sol, iterations=pt_iters)

            # Ultra SA
            sol = ultra_sa(sol, iterations=sa_iters)

            # Compaction
            sol = aggressive_compact(sol, 40000)
            sol = squeeze_boundary(sol, 15000)
            sol = optimize_rotations(sol, 15000)

            # Final SA
            sol = ultra_sa(sol, iterations=sa_iters // 2, temp_start=1.0)

            # Final compact
            sol = aggressive_compact(sol, 20000)

            if not is_valid(sol):
                sol = repair_solution(sol)
                if not is_valid(sol):
                    continue

            score = get_bounding_side(sol)
            if score < best_score:
                best_score = score
                best_sol = sol

        if best_sol is None:
            if n - 1 in solutions:
                prev = solutions[n - 1]
                prev_polys = [make_polygon(*p) for p in prev]
                idx = STRtree(prev_polys)
                new_tree = place_radially(prev_polys, idx)
                best_sol = prev + [new_tree]
            else:
                best_sol = build_solution(n, 'spiral', 0.55)

        best_sol = center_solution(best_sol)
        solutions[n] = best_sol
        scores[n] = get_bounding_side(best_sol)

        if verbose and (n % 10 == 0 or n <= 5):
            elapsed = time.time() - start
            total = sum((scores[i] ** 2) / i for i in range(1, n + 1))
            print(f"[Seed {seed}] n={n:3d}: side={scores[n]:.4f}, score={total:.2f}, time={elapsed:.0f}s")

    total_score = sum((scores[n] ** 2) / n for n in range(1, max_n + 1))

    if verbose:
        print(f"[Seed {seed}] Final score: {total_score:.4f}")

    return total_score, solutions

def run_parallel(seeds: List[int], max_n: int = 200, num_cores: int = None) -> Tuple[float, Dict[int, Solution]]:
    """Run multiple seeds in parallel."""
    if num_cores is None:
        num_cores = min(len(seeds), mp.cpu_count())

    print(f"Running {len(seeds)} seeds on {num_cores} cores")
    print("=" * 60)

    start = time.time()

    # Create pool and run
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
    print("=" * 60)
    print(f"Best score: {best_score:.4f}")
    print(f"Total time: {elapsed/3600:.2f} hours")

    return best_score, best_solutions

def main():
    parser = argparse.ArgumentParser(description="Parallel Tree Packing Solver")
    parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores")
    parser.add_argument("--seeds", type=int, default=8, help="Number of seeds to try")
    parser.add_argument("--max-n", type=int, default=200, help="Maximum n")
    args = parser.parse_args()

    print("=" * 60)
    print("PARALLEL TREE PACKING SOLVER")
    print("=" * 60)

    seeds = [42, 123, 456, 789, 2025, 1337, 7777, 9999][:args.seeds]

    best_score, best_solutions = run_parallel(seeds, args.max_n, args.cores)

    if best_solutions:
        valid, errors = validate_all(best_solutions)
        print(f"Valid: {valid}")

        if valid:
            create_submission(best_solutions, "submission.csv")
            print("Saved: submission.csv")

if __name__ == "__main__":
    main()
