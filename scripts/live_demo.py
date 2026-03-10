#!/usr/bin/env python3
"""
Démo temps réel — FDTD + Optimisation Génétique.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.viz.dashboard import main

if __name__ == "__main__":
    main()
