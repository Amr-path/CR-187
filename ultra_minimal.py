#!/usr/bin/env python3
"""
ultra_minimal.py - Ultra-minimal Tree Packing Solver

Fastest possible solver - minimal optimization, just ensure valid placement.
Goal: Complete all 200 trees as fast as possible.
"""

import math
import random
import time

from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# Tree polygon
TREE = Polygon([
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
])

def mp(x, y, d):
    p = TREE
    if d != 0:
        p = affinity.rotate(p, d, origin=(0, 0))
    if x != 0 or y != 0:
        p = affinity.translate(p, xoff=x, yoff=y)
    return p

def col(p, oth):
    for o in oth:
        if p.intersects(o) and not p.touches(o):
            return True
    return False

def colidx(p, idx, polys):
    for i in idx.query(p):
        if p.intersects(polys[i]) and not p.touches(polys[i]):
            return True
    return False

def sd(pls):
    if not pls:
        return 0.0
    ps = [mp(*p) for p in pls]
    b = unary_union(ps).bounds
    return max(b[2] - b[0], b[3] - b[1])

def sdp(ps):
    b = unary_union(ps).bounds
    return max(b[2] - b[0], b[3] - b[1])

def ctr(pls):
    ps = [mp(*p) for p in pls]
    b = unary_union(ps).bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in pls]

def ovlp(pls):
    ps = [mp(*p) for p in pls]
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            if ps[i].intersects(ps[j]) and not ps[i].touches(ps[j]):
                return True
    return False

# Fermat spiral
def spr(n, sc=0.52):
    g = math.pi * (3.0 - math.sqrt(5.0))
    return [(sc * math.sqrt(i) * math.cos(i * g),
             sc * math.sqrt(i) * math.sin(i * g)) for i in range(n)]

# Fast place
def plc(ps, idx, att=15):
    if not ps:
        return (0.0, 0.0, 0.0)

    for _ in range(att):
        t = random.uniform(0, 2 * math.pi)
        vx, vy = math.cos(t), math.sin(t)

        # Quick binary search
        lo, hi = 0.0, 8.0
        while hi - lo > 0.1:
            mid = (lo + hi) / 2
            ok = False
            for rot in [0, 90, 180, 270]:
                p = mp(mid * vx, mid * vy, rot)
                if not colidx(p, idx, ps):
                    ok = True
                    break
            if ok:
                hi = mid
            else:
                lo = mid

        px, py = hi * vx, hi * vy
        for rot in [0, 45, 90, 135, 180, 225, 270, 315]:
            p = mp(px, py, rot)
            if not colidx(p, idx, ps):
                return (px, py, rot)

    # Fallback
    return (10.0, 0.0, 0.0)

# Build initial solution
def bld(n, sc=0.52):
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    pos = spr(n, sc)
    sol = []
    ps = []

    for px, py in pos:
        if not ps:
            sol.append((px, py, 0.0))
            ps.append(mp(px, py, 0.0))
        else:
            idx = STRtree(ps)
            found = None
            for rot in [0, 45, 90, 135, 180, 225, 270, 315]:
                p = mp(px, py, rot)
                if not colidx(p, idx, ps):
                    found = rot
                    break
            if found is not None:
                sol.append((px, py, found))
                ps.append(mp(px, py, found))
            else:
                t = plc(ps, idx)
                sol.append(t)
                ps.append(mp(*t))

    return sol

# Super quick SA - only 500 iterations
def sa(pls, its=500):
    n = len(pls)
    if n <= 1:
        return pls

    cur = list(pls)
    ps = [mp(*p) for p in cur]
    cs = sdp(ps)

    best = list(cur)
    bs = cs

    T = 0.5
    cool = 0.99
    sh = cs * 0.05

    for _ in range(its):
        i = random.randrange(n)
        x, y, d = cur[i]

        nx = x + random.gauss(0, sh)
        ny = y + random.gauss(0, sh)
        nd = (d + random.gauss(0, 20)) % 360

        np = mp(nx, ny, nd)
        oth = ps[:i] + ps[i+1:]

        if col(np, oth):
            T *= cool
            continue

        old = ps[i]
        ps[i] = np
        ns = sdp(ps)

        if ns <= cs or random.random() < math.exp(-(ns - cs) / T):
            cur[i] = (nx, ny, nd)
            cs = ns
            if ns < bs:
                bs = ns
                best = list(cur)
        else:
            ps[i] = old

        T *= cool

    return best

class UltraMinSolver:
    def __init__(self, seed=42, verbose=True):
        self.seed = seed
        self.verbose = verbose
        random.seed(seed)
        self.sols = {}
        self.scrs = {}

    def solve_n(self, n):
        if n == 0:
            return []
        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            self.sols[1] = sol
            self.scrs[1] = 1.0
            return sol

        # Build from previous
        if n - 1 in self.sols:
            prev = self.sols[n - 1]
            prev_ps = [mp(*p) for p in prev]
            idx = STRtree(prev_ps)
            nt = plc(prev_ps, idx, att=20)
            sol = prev + [nt]
        else:
            sol = bld(n, 0.52)

        # Minimal SA - scale with sqrt(n)
        sa_its = min(800, 200 + int(math.sqrt(n) * 50))
        sol = sa(sol, its=sa_its)

        sol = ctr(sol)
        self.sols[n] = sol
        self.scrs[n] = sd(sol)
        return sol

    def solve_all(self, max_n=200):
        start = time.time()

        if self.verbose:
            print(f"Ultra-Min Solver: n=1 to {max_n}")

        for n in range(1, max_n + 1):
            self.solve_n(n)

            if self.verbose and n % 50 == 0:
                elapsed = time.time() - start
                score = self.total()
                print(f"n={n:3d}: side={self.scrs[n]:.4f}, score={score:.2f}, t={elapsed:.1f}s")

        if self.verbose:
            print(f"Final: {self.total():.4f}, Time: {time.time() - start:.1f}s")

        return self.sols

    def total(self):
        t = 0.0
        for n, sol in self.sols.items():
            s = self.scrs.get(n, sd(sol))
            t += (s ** 2) / n
        return t

def create_sub(sols, path="submission.csv"):
    with open(path, "w") as f:
        f.write("id,x,y,deg\n")
        for n in range(1, 201):
            pos = sols[n]
            ps = [mp(*p) for p in pos]
            b = unary_union(ps).bounds
            mx, my = b[0], b[1]
            for idx, (x, y, d) in enumerate(pos):
                f.write(f"{n:03d}_{idx},s{x - mx:.6f},s{y - my:.6f},s{d:.6f}\n")
    return path

def validate(sols):
    for n in range(1, 201):
        if n not in sols or len(sols[n]) != n or ovlp(sols[n]):
            return False
    return True

def main():
    print("ULTRA-MINIMAL SOLVER")
    print("=" * 40)

    solver = UltraMinSolver(seed=42, verbose=True)
    sols = solver.solve_all(max_n=200)

    valid = validate(sols)
    print(f"Valid: {valid}")

    if valid:
        print(f"Score: {solver.total():.4f}")
        create_sub(sols, "submission.csv")
        print("Saved: submission.csv")

if __name__ == "__main__":
    main()
