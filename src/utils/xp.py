"""
Détection automatique du backend de calcul.

Priorité :
  1. CuPy  - GPU NVIDIA (CUDA)
  2. NumPy - CPU (fallback universel : macOS Apple Silicon, CI, etc.)

Usage dans les autres modules :
    from src.utils.xp import xp as np, to_numpy
"""

import numpy as _numpy

_cupy = None
GPU_AVAILABLE = False
BACKEND = "NumPy (CPU)"

try:
    import cupy as _cupy
    n_devices = _cupy.cuda.runtime.getDeviceCount()
    if n_devices > 0:
        xp = _cupy
        GPU_AVAILABLE = True
        BACKEND = f"CuPy (CUDA) - {n_devices} GPU(s)"
    else:
        xp = _numpy
except Exception:
    xp = _numpy


def to_numpy(arr):
    """Convertit un array (cupy ou numpy) en numpy array CPU."""
    if _cupy is not None and isinstance(arr, _cupy.ndarray):
        return arr.get()
    return _numpy.asarray(arr)


def report():
    """Affiche le backend actif."""
    try:
        from rich.console import Console as _Console
        _Console(highlight=False).print(
            f"[dim]·[/dim] Backend calcul : "
            f"[bold {'green' if GPU_AVAILABLE else 'yellow'}]{BACKEND}[/]"
        )
    except ImportError:
        print(f"· Backend calcul : {BACKEND}")
