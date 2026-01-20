"""
Santa 2025 Christmas Tree Packing Solver v4 - EXTREME MODE

EXTREME OPTIMIZATION ONLY - No lite options.
Optimized for 14-core VPS.

Features:
- Extended Parallel Tempering (16 replicas)
- CMA-ES + Differential Evolution
- Ultra-long Simulated Annealing (800K+ iterations)
- Exhaustive search for small n
- Multi-phase aggressive compaction
- 14-core parallel processing by default

Target: Score < 55
"""

from .geometry import (
    make_tree_polygon,
    transform_tree,
    compute_bounding_square_side,
    check_all_overlaps,
    center_placements,
)

from .packing import (
    PackingSolver,
    SolverConfig,
    OptimizationConfig,
)

from .validate import (
    validate_solution,
    validate_all_solutions,
    compute_score,
    print_score_summary,
)

from .io_utils import (
    create_submission,
    print_solution_summary,
    get_output_path,
)

__version__ = "4.0.0"
__mode__ = "EXTREME"
__default_cores__ = 14
