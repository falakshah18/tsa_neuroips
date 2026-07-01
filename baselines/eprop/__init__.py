# baselines/eprop/__init__.py
"""
E-prop Baseline
"""

from .eprop_snn import (
    EligibilityTrace,
    EpropLinear,
    EpropSNN,
    EpropSNN_Vision,
    get_eprop_model,
)

__all__ = [
    'EligibilityTrace',
    'EpropLinear',
    'EpropSNN',
    'EpropSNN_Vision',
    'get_eprop_model',
]