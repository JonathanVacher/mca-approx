# """Generates clean, parameterized PRISM CTMC model specification files."""


# def generate_prism_model(
#     config,
#     simulator
# ) -> str:
    
#     tau=0.5
#     sig=1.5
#     h=config.h
#     n_states=simulator.n_states
#     x_min=config.x_safe_min-config.h
#     safe_xmin=config.x_safe_min
#     safe_xmax=config.x_safe_max
#     init_state=simulator.start_idx

#     """Generates a PRISM Continuous-Time Markov Chain (CTMC) model string

#     based on a nodal generator discretization matrix.
#     """


#     return f"""ctmc

# // SDE parameters
# const double tau = {tau};
# const double sig = {sig};

# // Grid and State Space parameters
# const double h = {h};
# const int N = {n_states};
# const double x_min = {x_min};

# // Safety Set boundaries
# const double safe_min = {safe_xmin};
# const double safe_max = {safe_xmax};

# // The physical state space value 'x' for a given index 's'
# formula x_val = x_min + s * h;

# // Transition rates based on the nodal CTMC generator discretization
# formula rate_up   = (0.5 * sig * sig / (h * h)) - (x_val / (tau * 2.0 * h));
# formula rate_down = (0.5 * sig * sig / (h * h)) + (x_val / (tau * 2.0 * h));

# module MCA_Ornstein_Uhlenbeck

#     // State index
#     s : [0..N-1] init {init_state};

#     // Internal state transitions
#     [] s > 0 & s < N-1 -> rate_up   : (s'=s+1) + rate_down : (s'=s-1);
#     [] s = 0          -> rate_up   : (s'=0) + rate_down : (s'=0);
#     [] s = N-1        -> rate_down : (s'=N-1) + rate_up   : (s'=N-1);

# endmodule

# // Labeling safety criteria
# label "safe" = x_val >= safe_min & x_val <= safe_max;
# """






"""Generates PRISM CTMC model with precomputed rates and absorbing boundaries."""

import numpy as np


def generate_prism_model(config, simulator) -> str:
    """
    PRISM CTMC with:
    - general mu(x), sigma(x)
    - precomputed transition rates
    - absorbing boundary states at s=0 and s=N-1
    """

    h = config.h
    x_space = simulator.x_space
    n_states = simulator.n_states

    safe_xmin = config.x_safe_min
    safe_xmax = config.x_safe_max
    init_state = simulator.start_idx
    x_min=config.x_safe_min-config.h

    # ------------------------------------------------------------
    # PRECOMPUTE RATES
    # ------------------------------------------------------------
    rate_up = np.zeros(n_states)
    rate_down = np.zeros(n_states)

    for i, x in enumerate(x_space):
        mu = config.mu(x)
        sigma = config.sigma(x)

        a = 0.5 * sigma * sigma / (h * h)
        b = mu / (2.0 * h)

        rate_up[i] = max(a + b, 0.0)
        rate_down[i] = max(a - b, 0.0)

    # ------------------------------------------------------------
    # PRISM CONSTANTS
    # ------------------------------------------------------------
    rate_consts_up = "\n".join(
        [f"const double rate_up_{i} = {rate_up[i]:.10f};" for i in range(n_states)]
    )

    rate_consts_down = "\n".join(
        [f"const double rate_down_{i} = {rate_down[i]:.10f};" for i in range(n_states)]
    )

    # ------------------------------------------------------------
    # TRANSITIONS WITH TRUE ABSORPTION
    # ------------------------------------------------------------
    transitions = []

    for i in range(n_states):

        # ABSORBING STATES
        if i == 0 or i == n_states - 1:
            transitions.append(
                f"    [] s={i} -> (s'={i});"
            )
            continue

        # INTERIOR STATES
        transitions.append(
            f"    [] s={i} -> "
            f"rate_up_{i} : (s'={i+1}) + "
            f"rate_down_{i} : (s'={i-1});"
        )

    transitions_str = "\n".join(transitions)

    return f"""ctmc

// Grid
const double h = {h};
const int N = {n_states};
const double x_min = {x_min};


// Safety set
const double safe_min = {safe_xmin};
const double safe_max = {safe_xmax};

// Initial state
const int init_state = {init_state};

// State-space mapping
formula x_val = x_min + s * h;

// --------------------------------------------------
// PRECOMPUTED RATES FROM mu(x), sigma(x)
// --------------------------------------------------
{rate_consts_up}

{rate_consts_down}

module MCA_SDE

    s : [0..N-1] init init_state;

{transitions_str}

endmodule

// Safety label
label "safe" = x_val >= safe_min & x_val <= safe_max;
"""





def generate_pctl(config) -> str:
    """Generates a PRISM pctl properties to verify.
    """
    T_max = config.t_max
    return f"""
const double T_max = {T_max};

// 1. Path Safety Probability: 
// Computes the probability that the Markov Chain STAYS continuously 
// within the safe set [safe_min, safe_max] from time t=0 up to T_max.
P=? [ G<=T_max "safe" ]

// 2. Fixed-Horizon Target-Set Probability:
// Computes the probability that the Markov Chain IS CURRENTLY in the 
// safe set exactly at the final time T_max (Transient probability).
P=? [ true U[T_max,T_max] "safe" ]

// Find the strategy that MAXIMIZES the probability of staying safe
// Pmax=? [ G<=T_max "safe" ]


"""
