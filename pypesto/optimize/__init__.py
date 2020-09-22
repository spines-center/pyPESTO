"""
Optimize
========

Multistart optimization with support for various optimizers.
"""

from .options import OptimizeOptions
from .optimize import (
    minimize)
from .optimizer import (
    Optimizer,
    ScipyOptimizer,
    IpoptOptimizer,
    DlibOptimizer,
    PyswarmOptimizer,
    PyswarmsOptimizer,
    CmaesOptimizer)
from .result import OptimizerResult
