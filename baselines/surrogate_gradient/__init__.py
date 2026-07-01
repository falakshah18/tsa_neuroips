# baselines/surrogate_gradient/__init__.py
"""
Surrogate Gradient SNN Baseline
"""

from .surrogate_grad_snn import (
    SurrogateGradSNN,
    SurrogateGradSNN_SHD,
    get_surrogate_grad_model,
)

__all__ = [
    'SurrogateGradSNN',
    'SurrogateGradSNN_SHD',
    'get_surrogate_grad_model',
]