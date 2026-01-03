#!/usr/bin/env python3
"""
best_solver.py - Production Tree Packing Solver for Santa 2025

Implements advanced optimization techniques with balanced parameters:
1. Genetic Algorithm (GA) for global optimization
2. Basin Hopping for escaping local minima
3. Extended Simulated Annealing with reheat
4. Multi-stage compaction
5. Rotation optimization

Run: python best_solver.py
Output: submission.csv
"""

import math
import random
import time
import copy
from typing import List, Tuple, Dict, Optional

from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# ============================================================================
# TREE GEOMETRY
# ============================================================================

TREE_COORDS = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]
BASE_POLYGON = Polygon(TREE_COORDS)

Placement = Tuple[float, float, float]
Solution = List[Placement]

def mp(x: float, y: float, d: float) -> Polygon:
    p = BASE_POLYGON
    if d != 0:
        p = affinity.rotate(p, d, origin=(0, 0))
    if x != 0 or y != 0:
        p = affinity.translate(p, xoff=x, yoff=y)
    return p

def col(p: Polygon, others: List[Polygon]) -> bool:
    for o in others:
        if p.intersects(o) and not p.touches(o):
            return True
    return False

def colidx(p: Polygon, idx: STRtree, polys: List[Polygon]) -> bool:
    for i in idx.query(p):
        if p.intersects(polys[i]) and not p.touches(polys[i]):
            return True
    return False

def side(pls: Solution) -> float:
    if not pls:
        return 0.0
    ps = [mp(*p) for p in pls]
    b = unary_union(ps).bounds
    return max(b[2] - b[0], b[3] - b[1])

def sidep(ps: List[Polygon]) -> float:
    b = unary_union(ps).bounds
    return max(b[2] - b[0], b[3] - b[1])

def center(pls: Solution) -> Solution:
    ps = [mp(*p) for p in pls]
    b = unary_union(ps).bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in pls]

def overlaps(pls: Solution) -> int:
    ps = [mp(*p) for p in pls]
    cnt = 0
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            if ps[i].intersects(ps[j]) and not ps[i].touches(ps[j]):
                cnt += 1
    return cnt

def valid(pls: Solution) -> bool:
    return overlaps(pls) == 0

# ============================================================================
# PLACEMENT
# ============================================================================

def spiral(n: int, sc: float = 0.48) -> List[Tuple[float, float]]:
    g = math.pi * (3.0 - math.sqrt(5.0))
    return [(sc * math.sqrt(i) * math.cos(i * g),
             sc * math.sqrt(i) * math.sin(i * g)) for i in range(n)]

def place_radial(ps: List[Polygon], idx: Optional[STRtree], att: int = 50) -> Placement:
    if not ps:
        return (0.0, 0.0, random.uniform(0, 360))

    best = None
    best_r = 999

    for _ in range(att):
        while True:
            t = random.uniform(0, 2 * math.pi)
            if random.random() < abs(math.sin(2 * t)) + 0.15:
                break

        vx, vy = math.cos(t), math.sin(t)
        lo, hi = 0.0, 12.0

        while hi - lo > 0.02:
            mid = (lo + hi) / 2
            ok = False
            for rot in [0, 45, 90, 135, 180, 225, 270, 315]:
                p = mp(mid * vx, mid * vy, rot)
                if idx:
                    if not colidx(p, idx, ps):
                        ok = True
                        break
                else:
                    if not col(p, ps):
                        ok = True
                        break
            if ok:
                hi = mid
            else:
                lo = mid

        if hi < best_r:
            px, py = hi * vx, hi * vy
            for rot in range(0, 360, 15):
                p = mp(px, py, rot)
                if idx:
                    if not colidx(p, idx, ps):
                        best_r = hi
                        best = (px, py, rot)
                        break
                else:
                    if not col(p, ps):
                        best_r = hi
                        best = (px, py, rot)
                        break

    return best if best else (10.0, 0.0, 0.0)

def build(n: int, sc: float = 0.48) -> Solution:
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    pos = spiral(n, sc)
    sol = []
    ps = []

    for px, py in pos:
        if not ps:
            sol.append((px, py, 0.0))
            ps.append(mp(px, py, 0.0))
        else:
            idx = STRtree(ps)
            found = None
            for rot in range(0, 360, 15):
                p = mp(px, py, rot)
                if not colidx(p, idx, ps):
                    found = rot
                    break
            if found is not None:
                sol.append((px, py, found))
                ps.append(mp(px, py, found))
            else:
                t = place_radial(ps, idx)
                sol.append(t)
                ps.append(mp(*t))

    return sol

# ============================================================================
# SIMULATED ANNEALING WITH REHEAT
# ============================================================================

def sa_reheat(pls: Solution, iters: int, T0: float = 2.0, Tf: float = 1e-9,
              reheat_every: int = 5000, reheat_factor: float = 0.2) -> Solution:
    n = len(pls)
    if n <= 1:
        return pls

    cur = list(pls)
    ps = [mp(*p) for p in cur]
    cs = sidep(ps)

    best = list(cur)
    bs = cs

    T = T0
    cool = (Tf / T0) ** (1.0 / iters)
    sh = cs * 0.1
    rt = 40.0

    stag = 0

    for it in range(iters):
        # Reheat periodically
        if it > 0 and it % reheat_every == 0:
            T = max(T, T0 * reheat_factor * (1 - it / iters))
            sh = bs * 0.08
            rt = 35.0

        i = random.randrange(n)
        x, y, d = cur[i]

        r = random.random()
        if r < 0.45:
            nx = x + random.gauss(0, sh)
            ny = y + random.gauss(0, sh)
            nd = d
        elif r < 0.75:
            nx, ny = x, y
            nd = (d + random.gauss(0, rt)) % 360
        elif r < 0.92:
            nx = x + random.gauss(0, sh * 0.6)
            ny = y + random.gauss(0, sh * 0.6)
            nd = (d + random.gauss(0, rt * 0.6)) % 360
        else:
            nx = x + random.gauss(0, sh * 2)
            ny = y + random.gauss(0, sh * 2)
            nd = random.uniform(0, 360)

        np = mp(nx, ny, nd)
        oth = ps[:i] + ps[i+1:]

        if col(np, oth):
            T *= cool
            stag += 1
            continue

        old = ps[i]
        ps[i] = np
        ns = sidep(ps)
        delta = ns - cs

        if delta <= 0 or random.random() < math.exp(-delta / T):
            cur[i] = (nx, ny, nd)
            cs = ns
            stag = 0
            if ns < bs:
                bs = ns
                best = list(cur)
        else:
            ps[i] = old

        T *= cool

        if stag > 300:
            sh = max(0.003, sh * 0.95)
            rt = max(1.0, rt * 0.95)
            stag = 0

    return best

# ============================================================================
# GENETIC ALGORITHM
# ============================================================================

def ga_crossover(p1: Solution, p2: Solution) -> Solution:
    n = len(p1)
    if n <= 2:
        return list(p1)
    pt = random.randint(1, n - 1)
    child = []
    for i in range(n):
        if i < pt:
            child.append(p1[i])
        else:
            x, y, _ = p1[i]
            _, _, d = p2[i]
            child.append((x, y, d))
    return child

def ga_mutate(sol: Solution, rate: float = 0.25) -> Solution:
    n = len(sol)
    mut = list(sol)
    ps = [mp(*p) for p in mut]
    cs = sidep(ps)

    for i in range(n):
        if random.random() < rate:
            x, y, d = mut[i]
            nx = x + random.gauss(0, cs * 0.04)
            ny = y + random.gauss(0, cs * 0.04)
            nd = (d + random.gauss(0, 15)) % 360
            np = mp(nx, ny, nd)
            oth = ps[:i] + ps[i+1:]
            if not col(np, oth):
                mut[i] = (nx, ny, nd)
                ps[i] = np

    return mut

def genetic_algorithm(init: Solution, pop_size: int = 12, gens: int = 15,
                     mut_rate: float = 0.25, cross_rate: float = 0.7, elite: int = 2) -> Solution:
    n = len(init)
    if n <= 2:
        return init

    # Initialize population
    pop = [init]
    for _ in range(pop_size - 1):
        v = ga_mutate(list(init), rate=0.4)
        if valid(v):
            pop.append(v)
        else:
            pop.append(list(init))

    best_sol = init
    best_sc = side(init)

    for gen in range(gens):
        # Evaluate
        fitness = [(side(s) if valid(s) else 999, s) for s in pop]
        fitness.sort(key=lambda x: x[0])

        if fitness[0][0] < best_sc:
            best_sc = fitness[0][0]
            best_sol = list(fitness[0][1])

        # Selection
        new_pop = [list(f[1]) for f in fitness[:elite]]

        while len(new_pop) < pop_size:
            # Tournament
            t1 = random.sample(fitness, 3)
            p1 = min(t1, key=lambda x: x[0])[1]
            t2 = random.sample(fitness, 3)
            p2 = min(t2, key=lambda x: x[0])[1]

            if random.random() < cross_rate:
                child = ga_crossover(p1, p2)
            else:
                child = list(p1)

            child = ga_mutate(child, mut_rate)

            if valid(child):
                new_pop.append(child)
            else:
                new_pop.append(list(p1))

        pop = new_pop

    return best_sol

# ============================================================================
# BASIN HOPPING
# ============================================================================

def local_min(sol: Solution, iters: int = 1500) -> Solution:
    return sa_reheat(sol, iters, T0=0.4, Tf=1e-8, reheat_every=iters+1)

def basin_hop(init: Solution, n_iter: int = 8, temp: float = 0.8, step: float = 0.25) -> Solution:
    n = len(init)
    if n <= 2:
        return init

    cur = local_min(init, iters=2000)
    cs = side(cur)

    best = list(cur)
    bs = cs

    for _ in range(n_iter):
        # Hop
        cand = list(cur)
        ps = [mp(*p) for p in cand]

        for i in range(n):
            if random.random() < step:
                x, y, d = cand[i]
                nx = x + random.gauss(0, cs * 0.12)
                ny = y + random.gauss(0, cs * 0.12)
                nd = (d + random.gauss(0, 25)) % 360
                np = mp(nx, ny, nd)
                oth = ps[:i] + ps[i+1:]
                if not col(np, oth):
                    cand[i] = (nx, ny, nd)
                    ps[i] = np

        # Local minimize
        cand = local_min(cand, iters=2000)
        cands = side(cand)

        # Accept?
        delta = cands - cs
        if delta < 0 or random.random() < math.exp(-delta / temp):
            cur = cand
            cs = cands
            if cands < bs:
                bs = cands
                best = list(cand)

    return best

# ============================================================================
# COMPACTION
# ============================================================================

def compact(pls: Solution, iters: int) -> Solution:
    n = len(pls)
    if n <= 1:
        return pls

    cur = list(pls)
    ps = [mp(*p) for p in cur]
    cs = sidep(ps)

    for _ in range(iters):
        b = unary_union(ps).bounds
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

        i = random.randrange(n)
        x, y, d = cur[i]

        dx, dy = cx - x, cy - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.01:
            continue

        step = random.uniform(0.005, min(0.06, dist * 0.35))
        nx = x + step * dx / dist
        ny = y + step * dy / dist
        nd = d
        if random.random() < 0.2:
            nd = (d + random.gauss(0, 8)) % 360

        np = mp(nx, ny, nd)
        oth = ps[:i] + ps[i+1:]

        if col(np, oth):
            continue

        old = ps[i]
        ps[i] = np
        ns = sidep(ps)

        if ns <= cs:
            cur[i] = (nx, ny, nd)
            cs = ns
        else:
            ps[i] = old

    return cur

def squeeze(pls: Solution, iters: int) -> Solution:
    n = len(pls)
    if n <= 2:
        return pls

    cur = list(pls)
    ps = [mp(*p) for p in cur]

    for _ in range(iters):
        b = unary_union(ps).bounds
        minx, miny, maxx, maxy = b
        s = max(maxx - minx, maxy - miny)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

        # Find boundary trees
        boundary = []
        for i, (x, y, d) in enumerate(cur):
            pb = ps[i].bounds
            on_b = (abs(pb[0] - minx) < 0.03 or abs(pb[2] - maxx) < 0.03 or
                   abs(pb[1] - miny) < 0.03 or abs(pb[3] - maxy) < 0.03)
            if on_b:
                boundary.append(i)

        if not boundary:
            break

        i = random.choice(boundary)
        x, y, d = cur[i]

        dx, dy = cx - x, cy - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.01:
            continue

        step = random.uniform(0.005, 0.025)
        nx = x + step * dx / dist
        ny = y + step * dy / dist

        np = mp(nx, ny, d)
        oth = ps[:i] + ps[i+1:]

        if col(np, oth):
            continue

        old = ps[i]
        ps[i] = np
        ns = sidep(ps)

        if ns < s:
            cur[i] = (nx, ny, d)
        else:
            ps[i] = old

    return cur

def opt_rot(pls: Solution, iters: int) -> Solution:
    n = len(pls)
    if n <= 1:
        return pls

    cur = list(pls)
    ps = [mp(*p) for p in cur]
    cs = sidep(ps)

    for _ in range(iters):
        i = random.randrange(n)
        x, y, d = cur[i]
        nd = (d + random.gauss(0, 20)) % 360

        np = mp(x, y, nd)
        oth = ps[:i] + ps[i+1:]

        if col(np, oth):
            continue

        old = ps[i]
        ps[i] = np
        ns = sidep(ps)

        if ns < cs:
            cur[i] = (x, y, nd)
            cs = ns
        else:
            ps[i] = old

    return cur

def local_refine(pls: Solution, iters: int) -> Solution:
    n = len(pls)
    if n <= 1:
        return pls

    cur = list(pls)
    ps = [mp(*p) for p in cur]
    cs = sidep(ps)

    st = 0.008
    rt = 4.0

    for it in range(iters):
        i = random.randrange(n)
        x, y, d = cur[i]

        nx = x + random.gauss(0, st)
        ny = y + random.gauss(0, st)
        nd = (d + random.gauss(0, rt)) % 360

        np = mp(nx, ny, nd)
        oth = ps[:i] + ps[i+1:]

        if col(np, oth):
            continue

        old = ps[i]
        ps[i] = np
        ns = sidep(ps)

        if ns < cs:
            cur[i] = (nx, ny, nd)
            cs = ns
        else:
            ps[i] = old

        if it % 1500 == 0 and it > 0:
            st = max(0.002, st * 0.85)
            rt = max(0.5, rt * 0.85)

    return cur

# ============================================================================
# MAIN SOLVER
# ============================================================================

class BestSolver:
    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        random.seed(seed)
        self.sols: Dict[int, Solution] = {}
        self.scrs: Dict[int, float] = {}

    def solve_n(self, n: int) -> Solution:
        if n == 0:
            return []
        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            self.sols[1] = sol
            self.scrs[1] = 1.0
            return sol

        best_sol = None
        best_s = float('inf')

        # Try multiple strategies
        for restart in range(3):
            sc = 0.46 + restart * 0.03

            if n - 1 in self.sols and restart == 0:
                prev = self.sols[n - 1]
                prev_ps = [mp(*p) for p in prev]
                idx = STRtree(prev_ps)
                nt = place_radial(prev_ps, idx, att=60)
                sol = prev + [nt]
            else:
                random.seed(self.seed + n * 100 + restart * 17)
                sol = build(n, sc)

            if not valid(sol):
                sol = build(n, 0.55)
                if not valid(sol):
                    continue

            # Optimization iterations scale with n
            sa_iters = 8000 + n * 80
            ga_gens = max(8, 15 - n // 20)
            bh_iters = max(4, 10 - n // 25)
            cmp_iters = 3000 + n * 20
            loc_iters = 4000 + n * 25

            # Phase 1: Extended SA
            sol = sa_reheat(sol, sa_iters, T0=2.5, Tf=1e-9, reheat_every=3000)

            # Phase 2: GA (for moderate n)
            if 5 <= n <= 150:
                sol = genetic_algorithm(sol, pop_size=12, gens=ga_gens)

            # Phase 3: Basin Hopping
            if n >= 3:
                sol = basin_hop(sol, n_iter=bh_iters)

            # Phase 4: Compaction
            sol = compact(sol, cmp_iters)
            sol = squeeze(sol, cmp_iters // 2)

            # Phase 5: Rotation optimization
            sol = opt_rot(sol, cmp_iters // 2)

            # Phase 6: Local refinement
            sol = local_refine(sol, loc_iters)

            # Phase 7: Final SA
            sol = sa_reheat(sol, sa_iters // 2, T0=0.8, Tf=1e-10, reheat_every=sa_iters)

            # Phase 8: Final compact
            sol = compact(sol, cmp_iters // 3)

            if not valid(sol):
                continue

            s = side(sol)
            if s < best_s:
                best_s = s
                best_sol = sol

        if best_sol is None:
            if n - 1 in self.sols:
                prev = self.sols[n - 1]
                prev_ps = [mp(*p) for p in prev]
                idx = STRtree(prev_ps)
                nt = place_radial(prev_ps, idx)
                best_sol = prev + [nt]
            else:
                best_sol = build(n, 0.6)

        best_sol = center(best_sol)
        self.sols[n] = best_sol
        self.scrs[n] = side(best_sol)
        return best_sol

    def solve_all(self, max_n: int = 200) -> Dict[int, Solution]:
        start = time.time()

        if self.verbose:
            print(f"Best Solver: n=1 to {max_n}, seed={self.seed}")
            print("=" * 60)

        for n in range(1, max_n + 1):
            self.solve_n(n)

            if self.verbose and (n % 10 == 0 or n <= 5):
                elapsed = time.time() - start
                score = self.total()
                eta = (elapsed / n) * (max_n - n) if n > 0 else 0
                print(f"n={n:3d}: side={self.scrs[n]:.4f}, score={score:.2f}, "
                      f"time={elapsed:.0f}s, ETA={eta:.0f}s")

        if self.verbose:
            print("=" * 60)
            print(f"Final Score: {self.total():.4f}")
            print(f"Total time: {time.time() - start:.0f}s")

        return self.sols

    def total(self) -> float:
        t = 0.0
        for n, sol in self.sols.items():
            s = self.scrs.get(n, side(sol))
            t += (s ** 2) / n
        return t

# ============================================================================
# OUTPUT
# ============================================================================

def create_sub(sols: Dict[int, Solution], path: str = "submission.csv") -> str:
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

def validate_all(sols: Dict[int, Solution]) -> bool:
    for n in range(1, 201):
        if n not in sols or len(sols[n]) != n or not valid(sols[n]):
            return False
    return True

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("BEST TREE PACKING SOLVER")
    print("GA + Basin Hopping + Extended SA")
    print("=" * 60)

    best_score = float('inf')
    best_sols = None

    seeds = [42, 123, 456]

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")

        solver = BestSolver(seed=seed, verbose=True)
        sols = solver.solve_all(max_n=200)

        if not validate_all(sols):
            print("Validation failed!")
            continue

        score = solver.total()
        print(f"Score: {score:.4f}")

        if score < best_score:
            best_score = score
            best_sols = copy.deepcopy(sols)
            print(f"*** NEW BEST: {score:.4f} ***")
            # Save intermediate
            create_sub(best_sols, "submission.csv")

    if best_sols:
        print(f"\n{'='*60}")
        print(f"BEST SCORE: {best_score:.4f}")
        create_sub(best_sols, "submission.csv")
        print("Saved: submission.csv")
        print(f"Valid: {validate_all(best_sols)}")

if __name__ == "__main__":
    main()
