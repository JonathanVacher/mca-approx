"""Configuration module holding SDE model parameters and global state settings."""

import numpy as np


class SDEConfig:
    """Stores the parameter specifications of the Ornstein-Uhlenbeck SDE

    and grid parameters for continuous-time Markov chain approximation.
    """

    def __init__(
        self,
        dt: float = 0.05,
        t_max: float = 0.5,
        tau: float = 0.5,
        sig: float = 1.0,
        h: float = 0.005,
        # alpha: float = 0.5,
        # beta: float = 1.0,
        x_safe_min: float =-2.0,
        x_safe_max: float = 2.0,
        epsilon: float = 0.3,
        x_min: float = -2.5,
        x_max: float = 2.5,
        x_init: float = 0.5,
        seed: int = 5,
    ):
        self.dt = dt
        self.t_max = t_max
        self.tau = tau
        self.sig = sig
        self.h = h
        self.x_safe_min = x_safe_min
        self.x_safe_max = x_safe_max
        self.epsilon = epsilon
        self.x_min = x_min
        self.x_max = x_max
        self.x_init = x_init
        self.seed = seed

        # Enforce global random state reproducibly
        np.random.seed(seed)

    def drift_step(self, x: float) -> float:
        """Computes one Euler-Maruyama discretization iteration displacement."""
        noise = self.sig * np.sqrt(self.dt) * np.random.randn()
        return -x / self.tau * self.dt + noise