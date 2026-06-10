"""Configuration module holding controlled SDE model parameters and global state settings."""

import numpy as np
from typing import Callable, List, Optional


class SDEConfig:
    """Stores the parameter specifications of the (Controlled) SDE
    and grid parameters for continuous-time Markov chain approximation.
    """

    def __init__(
        self,
        mu: Callable[..., float],  # Can take (x) or (x, u)
        sigma: Callable[[float], float],
        dt: float = 0.05,
        t_max: float = 0.5,
        h: float = 0.005,
        x_safe_min: float = -2.0,
        x_safe_max: float = 2.0,
        epsilon: float = 0.3,
        x_min: float = -2.5,
        x_max: float = 2.5,
        x_init: float = 0.5,
        seed: int = 5,
        U: Optional[List[float]] = None,  # Discrete control actions, e.g., [-1.0, 1.0]
    ):
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.t_max = t_max
        self.h = h
        self.x_safe_min = x_safe_min
        self.x_safe_max = x_safe_max
        self.epsilon = epsilon
        self.x_min = x_min
        self.x_max = x_max
        self.x_init = x_init
        self.seed = seed
        self.U = U
        
        # Helper property to check if control is active
        self.is_controlled = U is not None