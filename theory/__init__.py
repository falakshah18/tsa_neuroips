# theory/__init__.py
"""
Theory package
"""

from .formulation import FormalProblemStatement
from .proofs import (
    prove_convergence,
    prove_energy_bounds,
    prove_complexity,
)

__all__ = [
    'FormalProblemStatement',
    'prove_convergence',
    'prove_energy_bounds',
    'prove_complexity',
]