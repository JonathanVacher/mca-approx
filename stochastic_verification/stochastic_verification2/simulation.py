"""Executes trajectory comparisons between controlled SDE steps and Markov chain models."""

from typing import Dict, Tuple
import numpy as np
from .config import SDEConfig
import matplotlib.pyplot as plt


class StochasticSimulator:
    """Manages structural simulation validation over approximated transition vectors."""

    def __init__(self, config: SDEConfig):
        self.config = config
        self.x_space = np.arange(config.x_safe_min - config.h, config.x_safe_max + 2*config.h, config.h)
        self.n_states = self.x_space.shape[0]
        self.start_idx = np.searchsorted(self.x_space, self.config.x_init)
        
    def run_sde_simulation(self, n_paths: int = 100) -> Dict[str, np.ndarray]:
        """Simulates SDE paths using Euler-Maruyama. 
        Selects a random control u at each step if control is active.
        """
        np.random.seed(self.config.seed)
        time_steps = np.arange(0, self.config.t_max, self.config.dt)
        n_steps = len(time_steps)
        
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = self.config.x_init
        
        for i in range(1, n_steps):
            x_prev = paths[:, i - 1]
            
            if self.config.is_controlled:
                # Pick a random control choice for each path at this time step
                u_choices = np.random.choice(self.config.U, size=n_paths)
                # Compute drift element-wise
                mu_val = np.array([self.config.mu(x, u) for x, u in zip(x_prev, u_choices)])
            else:
                mu_val = np.array([self.config.mu(x) for x in x_prev])
                
            sigma_val = np.array([self.config.sigma(x) for x in x_prev])
            
            dW = np.random.normal(0, np.sqrt(self.config.dt), size=n_paths)
            paths[:, i] = x_prev + mu_val * self.config.dt + sigma_val * dW
            
        return {"time_steps": time_steps, "sde_paths": paths}