#!/usr/bin/env python3
"""
quick_solver.py - Ultra-fast Tree Packing Solver

Designed to complete all 200 trees very quickly while still achieving good scores.
Uses efficient algorithms with minimal iterations.
"""

import math
import random
import time
from typing import List, Tuple, Dict, Optional

from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# Tree polygon (15 vertices)
TREE = Polygon([
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
])

def mkpoly(x, y, deg):
    p = TREE
    if deg != 0:
        p = affinity.rotate(p, deg, origin=(0, 0))
    if x != 0 or y != 0:
        p = affinity.translate(p, xoff=x, yoff=y)
    return p

def coll(poly, others):
    for o in others:
        if poly.intersects(o) and not poly.touches(o):
            return True
    return False

def coll_idx(poly, idx, polys):
    for i in idx.query(poly):
        if poly.intersects(polys[i]) and not poly.touches(polys[i]):
            return True
    return False

def side(placements):
    if not placements:
        return 0.0
    polys = [mkpoly(*p) for p in placements]
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])

def side_p(polys):
    if not polys:
        return 0.0
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])

def center(placements):
    if not placements:
        return placements
    polys = [mkpoly(*p) for p in placements]
    b = unary_union(polys).bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]

def overlaps(placements):
    polys = [mkpoly(*p) for p in placements]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False

# Fermat spiral
def spiral(n, sc=0.52):
    g = math.pi * (3.0 - math.sqrt(5.0))
    return [(sc * math.sqrt(i) * math.cos(i * g),
             sc * math.sqrt(i) * math.sin(i * g)) for i in range(n)]

# Place tree radially
def place(polys, idx, att=40):
    if not polys:
        return (0.0, 0.0, random.uniform(0, 360))

    best = None
    best_r = 999

    for _ in range(att):
        while True:
            t = random.uniform(0, 2 * math.pi)
            if random.random() < abs(math.sin(2 * t)) + 0.2:
                break

        vx, vy = math.cos(t), math.sin(t)
        lo, hi = 0.0, 12.0

        while hi - lo > 0.03:
            mid = (lo + hi) / 2
            ok = False
            for rot in [0, 90, 180, 270]:
                poly = mkpoly(mid * vx, mid * vy, rot)
                if idx:
                    c = coll_idx(poly, idx, polys)
                else:
                    c = coll(poly, polys)
                if not c:
                    ok = True
                    break
            if ok:
                hi = mid
            else:
                lo = mid

        if hi < best_r:
            px, py = hi * vx, hi * vy
            best_rot = 0.0
            for rot in range(0, 360, 30):
                poly = mkpoly(px, py, rot)
                if idx:
                    c = coll_idx(poly, idx, polys)
                else:
                    c = coll(poly, polys)
                if not c:
                    best_rot = float(rot)
                    break
            best_r = hi
            best = (px, py, best_rot)

    return best if best else (0.0, 0.0, 0.0)

# Build solution
def build(n, sc=0.52):
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    pos = spiral(n, sc)
    sol = []
    polys = []

    for px, py in pos:
        if not polys:
            rot = random.uniform(0, 360)
            sol.append((px, py, rot))
            polys.append(mkpoly(px, py, rot))
        else:
            idx = STRtree(polys)
            found = None
            for rot in range(0, 360, 30):
                poly = mkpoly(px, py, rot)
                if not coll_idx(poly, idx, polys):
                    found = rot
                    break
            if found is not None:
                sol.append((px, py, found))
                polys.append(mkpoly(px, py, found))
            else:
                p = place(polys, idx)
                sol.append(p)
                polys.append(mkpoly(*p))

    return sol

# Quick SA
def sa(placements, its=5000, T0=1.0, Tf=1e-6):
    n = len(placements)
    if n <= 1:
        return placements

    curr = list(placements)
    polys = [mkpoly(*p) for p in curr]
    curr_s = side_p(polys)

    best = list(curr)
    best_s = curr_s

    T = T0
    cool = (Tf / T0) ** (1.0 / its)
    sh = curr_s * 0.1
    rt = 40.0

    for it in range(its):
        i = random.randrange(n)
        x, y, deg = curr[i]

        r = random.random()
        if r < 0.5:
            nx = x + random.gauss(0, sh)
            ny = y + random.gauss(0, sh)
            nd = deg
        elif r < 0.8:
            nx, ny = x, y
            nd = (deg + random.gauss(0, rt)) % 360
        else:
            nx = x + random.gauss(0, sh * 0.6)
            ny = y + random.gauss(0, sh * 0.6)
            nd = (deg + random.gauss(0, rt * 0.6)) % 360

        np = mkpoly(nx, ny, nd)
        oth = polys[:i] + polys[i+1:]

        if coll(np, oth):
            T *= cool
            continue

        old = polys[i]
        polys[i] = np
        new_s = side_p(polys)
        d = new_s - curr_s

        if d <= 0 or random.random() < math.exp(-d / T):
            curr[i] = (nx, ny, nd)
            curr_s = new_s
            if new_s < best_s:
                best_s = new_s
                best = list(curr)
        else:
            polys[i] = old

        T *= cool

    return best

# Quick compact
def compact(placements, its=1000):
    n = len(placements)
    if n <= 1:
        return placements

    curr = list(placements)
    polys = [mkpoly(*p) for p in curr]
    curr_s = side_p(polys)

    for _ in range(its):
        b = unary_union(polys).bounds
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

        i = random.randrange(n)
        x, y, deg = curr[i]

        dx, dy = cx - x, cy - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.01:
            continue

        step = random.uniform(0.005, min(0.05, dist * 0.3))
        nx = x + step * dx / dist
        ny = y + step * dy / dist

        np = mkpoly(nx, ny, deg)
        oth = polys[:i] + polys[i+1:]

        if coll(np, oth):
            continue

        old = polys[i]
        polys[i] = np
        new_s = side_p(polys)

        if new_s <= curr_s:
            curr[i] = (nx, ny, deg)
            curr_s = new_s
        else:
            polys[i] = old

    return curr

class QuickSolver:
    def __init__(self, seed=42, verbose=True):
        self.seed = seed
        self.verbose = verbose
        random.seed(seed)
        self.solutions = {}
        self.scores = {}

    def solve_n(self, n):
        if n == 0:
            return []
        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            self.solutions[1] = sol
            self.scores[1] = side(sol)
            return sol

        # Build from previous
        if n - 1 in self.solutions:
            prev = self.solutions[n - 1]
            prev_polys = [mkpoly(*p) for p in prev]
            idx = STRtree(prev_polys)
            new_tree = place(prev_polys, idx, att=50)
            sol = prev + [new_tree]
        else:
            sol = build(n, 0.52)

        if overlaps(sol):
            sol = build(n, 0.55)

        # Quick optimization - scale with n
        sa_its = min(8000, 3000 + n * 25)
        comp_its = min(2000, 500 + n * 8)

        sol = sa(sol, its=sa_its)
        sol = compact(sol, its=comp_its)
        sol = sa(sol, its=sa_its // 2, T0=0.3)

        if overlaps(sol):
            # Fallback
            if n - 1 in self.solutions:
                prev = self.solutions[n - 1]
                prev_polys = [mkpoly(*p) for p in prev]
                idx = STRtree(prev_polys)
                new_tree = place(prev_polys, idx)
                sol = prev + [new_tree]
            else:
                sol = build(n, 0.6)

        sol = center(sol)
        self.solutions[n] = sol
        self.scores[n] = side(sol)
        return sol

    def solve_all(self, max_n=200):
        start = time.time()

        if self.verbose:
            print(f"Quick Solver: n=1 to {max_n}")
            print("=" * 50)

        for n in range(1, max_n + 1):
            self.solve_n(n)

            if self.verbose and (n % 25 == 0 or n <= 3):
                elapsed = time.time() - start
                score = self.total()
                print(f"n={n:3d}: side={self.scores[n]:.4f}, score={score:.2f}, t={elapsed:.1f}s")

        if self.verbose:
            print("=" * 50)
            print(f"Final: {self.total():.4f}")
            print(f"Time: {time.time() - start:.1f}s")

        return self.solutions

    def total(self):
        t = 0.0
        for n, sol in self.solutions.items():
            s = self.scores.get(n, side(sol))
            t += (s ** 2) / n
        return t

def create_sub(solutions, path="submission.csv"):
    with open(path, "w") as f:
        f.write("id,x,y,deg\n")
        for n in range(1, 201):
            positions = solutions[n]
            polys = [mkpoly(*p) for p in positions]
            b = unary_union(polys).bounds
            mx, my = b[0], b[1]
            for idx, (x, y, deg) in enumerate(positions):
                f.write(f"{n:03d}_{idx},s{x - mx:.6f},s{y - my:.6f},s{deg:.6f}\n")
    return path

def validate(solutions):
    for n in range(1, 201):
        if n not in solutions:
            return False, f"Missing n={n}"
        if len(solutions[n]) != n:
            return False, f"n={n}: wrong count"
        if overlaps(solutions[n]):
            return False, f"n={n}: overlaps"
    return True, None

def main():
    print("=" * 50)
    print("QUICK TREE PACKING SOLVER")
    print("=" * 50)

    best_score = float('inf')
    best_solutions = None

    for seed in [42, 123]:
        print(f"\nSeed: {seed}")
        print("-" * 30)

        solver = QuickSolver(seed=seed, verbose=True)
        solutions = solver.solve_all(max_n=200)

        valid, err = validate(solutions)
        if not valid:
            print(f"Error: {err}")
            continue

        score = solver.total()
        if score < best_score:
            best_score = score
            best_solutions = solutions
            print(f"*** Best: {score:.4f} ***")

    if best_solutions:
        print(f"\n{'='*50}")
        print(f"BEST SCORE: {best_score:.4f}")
        path = create_sub(best_solutions, "submission.csv")
        print(f"Saved: {path}")

        valid, _ = validate(best_solutions)
        print(f"Valid: {valid}")

if __name__ == "__main__":
    main()
