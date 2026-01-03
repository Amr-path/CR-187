#!/usr/bin/env python3
"""
run_solver.py - Single seed solver run
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

# Fermat spiral - tight packing
def spr(n, sc=0.48):
    g = math.pi * (3.0 - math.sqrt(5.0))
    return [(sc * math.sqrt(i) * math.cos(i * g),
             sc * math.sqrt(i) * math.sin(i * g)) for i in range(n)]

# Fast radial placement
def plc(ps, idx, att=25):
    if not ps:
        return (0.0, 0.0, 0.0)

    best = None
    best_r = 999

    for _ in range(att):
        while True:
            t = random.uniform(0, 2 * math.pi)
            if random.random() < abs(math.sin(2 * t)) + 0.15:
                break

        vx, vy = math.cos(t), math.sin(t)
        lo, hi = 0.0, 8.0

        while hi - lo > 0.04:
            mid = (lo + hi) / 2
            ok = False
            for rot in [0, 60, 120, 180, 240, 300]:
                p = mp(mid * vx, mid * vy, rot)
                if not colidx(p, idx, ps):
                    ok = True
                    break
            if ok:
                hi = mid
            else:
                lo = mid

        if hi < best_r:
            px, py = hi * vx, hi * vy
            for rot in range(0, 360, 20):
                p = mp(px, py, rot)
                if not colidx(p, idx, ps):
                    best_r = hi
                    best = (px, py, rot)
                    break

    return best if best else (8.0, 0.0, 0.0)

# Build initial solution
def bld(n, sc=0.48):
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
            for rot in range(0, 360, 20):
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

# Quick SA
def sa(pls, its=800, T0=0.8):
    n = len(pls)
    if n <= 1:
        return pls

    cur = list(pls)
    ps = [mp(*p) for p in cur]
    cs = sdp(ps)

    best = list(cur)
    bs = cs

    T = T0
    cool = (1e-5 / T0) ** (1.0 / its)
    sh = cs * 0.06
    rt = 25.0

    for _ in range(its):
        i = random.randrange(n)
        x, y, d = cur[i]

        r = random.random()
        if r < 0.5:
            nx = x + random.gauss(0, sh)
            ny = y + random.gauss(0, sh)
            nd = d
        elif r < 0.8:
            nx, ny = x, y
            nd = (d + random.gauss(0, rt)) % 360
        else:
            nx = x + random.gauss(0, sh * 0.5)
            ny = y + random.gauss(0, sh * 0.5)
            nd = (d + random.gauss(0, rt * 0.5)) % 360

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

# Compact
def cmp(pls, its=300):
    n = len(pls)
    if n <= 1:
        return pls

    cur = list(pls)
    ps = [mp(*p) for p in cur]
    cs = sdp(ps)

    for _ in range(its):
        b = unary_union(ps).bounds
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

        i = random.randrange(n)
        x, y, d = cur[i]

        dx, dy = cx - x, cy - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.01:
            continue

        step = random.uniform(0.01, min(0.05, dist * 0.3))
        nx = x + step * dx / dist
        ny = y + step * dy / dist

        np = mp(nx, ny, d)
        oth = ps[:i] + ps[i+1:]

        if col(np, oth):
            continue

        old = ps[i]
        ps[i] = np
        ns = sdp(ps)

        if ns <= cs:
            cur[i] = (nx, ny, d)
            cs = ns
        else:
            ps[i] = old

    return cur

def solve(seed=42):
    random.seed(seed)
    sols = {}
    scrs = {}

    start = time.time()
    print(f"Solver: seed={seed}, n=1 to 200")

    for n in range(1, 201):
        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            sols[1] = sol
            scrs[1] = 1.0
            continue

        # Build from previous
        if n - 1 in sols:
            prev = sols[n - 1]
            prev_ps = [mp(*p) for p in prev]
            idx = STRtree(prev_ps)
            nt = plc(prev_ps, idx, att=30)
            sol = prev + [nt]
        else:
            sol = bld(n, 0.48)

        if ovlp(sol):
            sol = bld(n, 0.52)

        # Quick optimization
        sa_its = min(1500, 500 + int(math.sqrt(n) * 70))
        cmp_its = min(500, 150 + int(math.sqrt(n) * 25))

        sol = sa(sol, its=sa_its)
        sol = cmp(sol, its=cmp_its)
        sol = sa(sol, its=sa_its // 2, T0=0.3)

        if ovlp(sol):
            if n - 1 in sols:
                prev = sols[n - 1]
                prev_ps = [mp(*p) for p in prev]
                idx = STRtree(prev_ps)
                nt = plc(prev_ps, idx)
                sol = prev + [nt]
            else:
                sol = bld(n, 0.55)

        sol = ctr(sol)
        sols[n] = sol
        scrs[n] = sd(sol)

        if n % 25 == 0:
            elapsed = time.time() - start
            total = sum((scrs[i] ** 2) / i for i in range(1, n + 1))
            print(f"n={n:3d}: side={scrs[n]:.4f}, score={total:.2f}, t={elapsed:.1f}s")

    total = sum((scrs[i] ** 2) / i for i in range(1, 201))
    print(f"Final: {total:.4f}, Time: {time.time() - start:.1f}s")

    return sols, total

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

if __name__ == "__main__":
    print("TREE PACKING SOLVER")
    print("=" * 40)

    sols, score = solve(seed=42)

    valid = validate(sols)
    print(f"Valid: {valid}")

    if valid:
        create_sub(sols, "submission.csv")
        print(f"Score: {score:.4f}")
        print("Saved: submission.csv")
