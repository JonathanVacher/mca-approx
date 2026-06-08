"""Performs analytical C^4 smooth gluing over working boundaries using SymPy."""

from typing import Tuple
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from .config import SDEConfig


class PolynomialGluing:
    """Solves algebraic equations to seamlessly transition smooth indicators

    across stabilization and operational margins.
    """

    def __init__(self, config: SDEConfig, degree: int = 9):
        self.config = config
        self.degree = degree
        self.num_equations = degree + 1
        self.x_safe_min = config.x_safe_min
        self.x_safe_max = config.x_safe_max
        self.epsilon = config.epsilon
        self.x_min = config.x_min
        self.x_max = config.x_max     
        # self.alpha = config.alpha
        # self.beta = config.beta

        self._build_and_solve()

    def _build_and_solve(self):
        """Constructs symbolic derivative constraint systems and evaluates smooth coefficients."""
        symbols = sp.symbols(f"a0:{self.num_equations}")
        x = sp.symbols("x")

        # Build boundary derivative constraint matrices
        eq_tab = [0] * self.num_equations
        for ii in range(int(self.num_equations / 2)):
            for jj in range(self.num_equations):
                factor = 1
                for kk in range(-ii + 1, 1):
                    factor *= jj + kk
                eq_tab[ii] += factor * symbols[jj] * x ** (jj - ii)

        eq_tab[0] -= 1  # Boundary target identity step

        # Swap tracking constraints over limits
        eq_tab_beta = [0] * self.num_equations
        for ii in range(int(self.num_equations / 2)):
            for jj in range(self.num_equations):
                factor = 1
                for kk in range(-ii + 1, 1):
                    factor *= jj + kk
                idx = ii + int(self.num_equations / 2)
                eq_tab_beta[idx] += factor * symbols[jj] * x ** (jj - ii)

        # Substitute operational domain bounds
        final_systemd = []
        for lines in range(int(self.num_equations / 2)):
            final_systemd.append(eq_tab[lines].subs(x, self.x_safe_max))
            final_systemd.append(eq_tab_beta[lines + int(self.num_equations / 2)].subs(x, self.x_safe_max+self.epsilon))

        final_systemg = []
        for lines in range(int(self.num_equations / 2)):
            final_systemg.append(eq_tab[lines].subs(x, self.x_safe_min))
            final_systemg.append(eq_tab_beta[lines + int(self.num_equations / 2)].subs(x, self.x_safe_min-self.epsilon))


        # Solve system linearly
        solved_coeffsg = sp.solve(final_systemg, symbols)
        polynomial_funcg = sum(solved_coeffsg[symbols[i]] * x**i for i in range(self.num_equations))
        solved_coeffsd = sp.solve(final_systemd, symbols)
        polynomial_funcd = sum(solved_coeffsd[symbols[i]] * x**i for i in range(self.num_equations))


        # Vectorize operations for crisp matplotlib analysis
        self.p_funcg = sp.lambdify(x, polynomial_funcg, "numpy")
        self.d3_funcg = sp.lambdify(x, sp.diff(polynomial_funcg, x, 3), "numpy")
        self.d4_funcg = sp.lambdify(x, sp.diff(polynomial_funcg, x, 4), "numpy")
        self.p_funcd = sp.lambdify(x, polynomial_funcd, "numpy")
        self.d3_funcd = sp.lambdify(x, sp.diff(polynomial_funcd, x, 3), "numpy")
        self.d4_funcd = sp.lambdify(x, sp.diff(polynomial_funcd, x, 4), "numpy")

        self._generate_profile_space()

    def _generate_profile_space(self):
        """Applies spatial mapping across intervals."""
        self.xx = np.linspace(self.config.x_min, self.config.x_max, 2000)
        self.pfxx = np.zeros_like(self.xx)

        # Vectorized assignment filtering over piecewise ranges
        mask1 = (self.xx >= self.x_safe_min) & (self.xx <= self.x_safe_max)
        mask2 = (self.xx > self.x_safe_max) & (self.xx <= self.x_safe_max+self.epsilon)
        mask3 = (self.xx >= self.x_safe_min-self.epsilon) & (self.xx < self.x_safe_min)
        # mask4 = ...
        # print(mask3)

        self.pfxx[mask1] = 1.0
        self.pfxx[mask2] = self.p_funcd(self.xx[mask2])
        self.pfxx[mask3] = self.p_funcg(self.xx[mask3])

        # Derivatives profiles over mapping zones
        self.pfd3xx = np.zeros_like(self.xx)
        self.pfd4xx = np.zeros_like(self.xx)
        self.pfd3xx[mask2] = self.d3_funcd(self.xx[mask2])
        self.pfd4xx[mask2] = self.d4_funcd(self.xx[mask2])
        self.pfd3xx[mask3] = -self.d3_funcg(self.xx[mask3])  # chain rule correction
        self.pfd4xx[mask3] = self.d4_funcg(self.xx[mask3])

    def plot_gluing_profile(self) -> Tuple[plt.Figure, plt.Axes]:
        """Plots the resulting C^4 smooth gluing function profile."""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.xx, self.pfxx, label="Gluing Function $\phi(x)$", color="purple", lw=2)

        ax.axvspan(self.x_safe_min, self.x_safe_max, color="blue", alpha=0.15, label="Indicator Zone")
        ax.axvspan(self.x_safe_min-self.epsilon, self.x_safe_min, color="red", alpha=0.15, label="$C^4$ Gluing Zone")
        ax.axvspan(self.x_safe_max, self.x_safe_max+self.epsilon, color="red", alpha=0.15)

        ax.set_xlabel("State Space ($x$)")
        ax.set_ylabel("Weight")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.6)
        return fig, ax

    def compute_error_bounds(self) -> Tuple[float,float]:
        """Evaluates weak approximation truncation error convergence margins."""
        n_t_steps = self.config.t_max / self.config.dt
        nu = self.config.sig
        theta = 1/self.config.tau
        # print(max(self.pfd4xx))
        # weak_bound = n_t_steps * self.config.dt * self.config.h  # Keeping exact original formula logic
        weak_bound = self.config.t_max*self.config.h**2*((nu**2)/8*max(self.pfd4xx)+ theta/6*max(self.pfd4xx)) # A VERIFIER

        lambdamin = self.config.sig**2
        Bmax = 1/self.config.tau

        density_bound = 1/((2*np.pi*lambdamin*self.config.t_max)**(1/2))*np.exp(Bmax**2*self.config.t_max/(2*lambdamin))
        smoothing_error = density_bound*(self.epsilon)/2

        return (float(weak_bound),float(smoothing_error))


        #         h = self.model.h
        # dt = self.model.dt
        # n_t_step = self.model.Tmax/dt

        # tau = self.model.tau
        # sig = self.model.sig

        # nu = sig # ????
        # theta = 1/tau # ????

        # # print(weakbound)

        # lambdamin = sig*sig
        # Bmax = 1/tau


        # densitybound = 1/((2*np.pi*lambdamin*n_t_step*dt)**(1/2))*np.exp(Bmax**2*n_t_step*dt/(2*lambdamin))
        # print("density bound:",densitybound)
        # # densitybound*(self.beta-self.alpha)/2)

        # smoothingbound = densitybound*(self.beta-self.alpha)/2

        # return (weakbound, smoothingbound)