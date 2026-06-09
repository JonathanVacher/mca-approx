"""Executes trajectory comparisons between SDE steps and Markov chain models."""

from typing import Dict, Tuple
import numpy as np
import scipy.linalg as linalg
import scipy.stats as st
from .config import SDEConfig


class StochasticSimulator:
    """Manages structural simulation validation over approximated transition vectors."""

    def __init__(self, config: SDEConfig):
        self.config = config
        # un etat supplémentaire en haut et en bas : cemetery states (absorbing)
        self.x_space = np.arange(config.x_safe_min - config.h, config.x_safe_max + 2*config.h, config.h)
        self.n_states = self.x_space.shape[0]
        self.q_matrix = self._assemble_generator()
        self.start_idx = np.searchsorted(self.x_space, self.config.x_init)

    def _assemble_generator(self) -> np.ndarray:
        """Assembles continuous time Markov Chain infinitesimal matrix generators."""
        h = self.config.h
        sig = self.config.sig
        tau = self.config.tau

        # Optimized vectorized generation
        diag_val = - (sig**2) / (h**2)
        q = np.diag(np.ones(self.n_states) * diag_val)

        for i in range(self.n_states):
            x_val = self.x_space[i]
            rate_up = (0.5 * sig**2 / h**2) - (x_val / (2.0 * tau * h))
            rate_down = (0.5 * sig**2 / h**2) + (x_val / (2.0 * tau * h))

            if i > 0:
                q[i, i - 1] = rate_down
            if i < self.n_states - 1:
                q[i, i + 1] = rate_up

        # Boundary condition adjustments matching initial structure
        # New BC : absorbing 
        q[0, :] = 0 #- (sig**2 / h**2)
        q[-1, :] = 0 #- (sig**2 / h**2)
        return q

    def run_monte_carlo(self, n_reps: int = 1000) -> Dict[str, np.ndarray]:
        """Runs paired SDE and Markov chain simulations for statistical validation."""
        dt = self.config.dt
        t_max = self.config.t_max
        n_steps = int(t_max / dt) + 2

        exp_dt_q = linalg.expm(dt * self.q_matrix).T
        # start_idx =  np.searchsorted(self.x_space, self.config.x_init) #(self.n_states - 1) // 2

        sde_paths = np.zeros((n_reps, n_steps))
        markov_paths = np.zeros((n_reps, n_steps))

        for rep in range(n_reps):
            t = 0.0
            step_idx = 0

            # State space trajectory initial conditions
            x_sde = self.config.x_init
            markov_state_vector = np.zeros(self.n_states)
            markov_state_vector[self.start_idx] = 1.0

            sde_paths[rep, step_idx] = x_sde
            markov_paths[rep, step_idx] = self.x_space[self.start_idx]

            while t <= t_max:
                t += dt
                step_idx += 1

                # 1. Update SDE path
                rdnum = np.random.randn()
                noise = self.config.sig * np.sqrt(dt) * rdnum
                x_sde = (1 - dt / self.config.tau) * x_sde + noise
                sde_paths[rep, step_idx] = x_sde

                # 2. Update Markov chain path
                next_prob_dist = exp_dt_q @ markov_state_vector
                next_prob_dist = next_prob_dist/np.sum(next_prob_dist)
                draw_unif = st.norm.cdf(rdnum)
                
                # Cumulative transition distribution mapping
                # draw_states = np.cumsum(draw_unif<np.cumsum(M_))==1
                cum_sum_dist = np.cumsum(next_prob_dist)
                state_idx = np.searchsorted(cum_sum_dist,draw_unif)
                if state_idx >= self.n_states:
                    state_idx = self.n_states - 1

                #FIX HERE : make absorbing

                # state_idx = np.searchsorted(np.cumsum(next_prob_dist), draw_unif)
                state_idx = min(max(state_idx, 0), self.n_states - 1)


                markov_state_vector = np.zeros(self.n_states)
                markov_state_vector[state_idx] = 1.0
                # print(state_idx)
                markov_paths[rep, step_idx] = self.x_space[state_idx]

        return {"sde": sde_paths, "markov": markov_paths, "time_steps": np.arange(n_steps) * dt}