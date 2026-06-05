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
        alpha: float = 0.5,
        beta: float = 1.0,
        x_max: float = 1.2,
        seed: int = 5,
    ):
        self.dt = dt
        self.t_max = t_max
        self.tau = tau
        self.sig = sig
        self.h = h
        self.alpha = alpha
        self.beta = beta
        self.x_max = x_max
        self.seed = seed

        # Enforce global random state reproducibly
        np.random.seed(seed)

    def drift_step(self, x: float) -> float:
        """Computes one Euler-Maruyama discretization iteration displacement."""
        noise = self.sig * np.sqrt(self.dt) * np.random.randn()
        return -x / self.tau * self.dt + noise