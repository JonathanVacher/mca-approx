"""Generates PRISM CTMC/MDP model with precomputed rates and absorbing boundaries."""

import numpy as np
import scipy.linalg as linalg


#CTMC generation 
def generate_prism_ctmc_model(config, simulator) -> str:
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
    # TRANSITIONS WITH ABSORBING BC
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



# MDP generation 
def generate_prism_mdp_model(config, simulator) -> str:
    """
    Generate a PRISM MDP whose transition probabilities are the
    precomputed matrix exponentials

        P_u = exp(dt * Q_u)

    used by the Monte-Carlo simulator.
    """

    if not config.is_controlled:
        raise ValueError(
            "MDP generation requires config.is_controlled=True"
        )

    h = config.h
    n_states = simulator.n_states
    x_min = config.x_safe_min - config.h

    safe_xmin = config.x_safe_min
    safe_xmax = config.x_safe_max
    init_state = simulator.start_idx

    # ----------------------------------------------------------
    # PRECOMPUTE TRANSITION MATRICES
    # ----------------------------------------------------------

    transition_matrices = []

    for q_u in simulator.q_matrices:
        P_u = linalg.expm(config.dt * q_u)

        # numerical cleanup
        P_u[P_u < 0] = 0.0
        P_u = P_u / P_u.sum(axis=1, keepdims=True)

        transition_matrices.append(P_u)

    # ----------------------------------------------------------
    # MODEL HEADER
    # ----------------------------------------------------------

    lines = [
        "mdp",
        "",
        f"const int N = {n_states};",
        f"const double h = {h};",
        f"const double x_min = {x_min};",
        "",
        f"const double safe_min = {safe_xmin};",
        f"const double safe_max = {safe_xmax};",
        "",
        f"const int init_state = {init_state};",
        "",
        "formula x_val = x_min + s*h;",
        "",
        "module MCA_SDE",
        "",
        "    s : [0..N-1] init init_state;",
        ""
    ]

    # ----------------------------------------------------------
    # TRANSITIONS
    # ----------------------------------------------------------

    tol = 1e-10

    for state in range(n_states):

        # absorbing boundaries
        if state == 0 or state == n_states - 1:

            for action_idx in range(len(config.U)):
                lines.append(
                    f"    [u{action_idx}] s={state} -> 1.0:(s'={state});"
                )

            lines.append("")
            continue

        # for action_idx, P_u in enumerate(transition_matrices):

        #     row = P_u[state]

        #     successors = []

        #     for target in range(n_states):

        #         p = row[target]

        #         if p > tol:
        #             successors.append(
        #                 f"{p:.16g}:(s'={target})"
        #             )

        #     lines.append(
        #         f"    [u{action_idx}] s={state} -> "
        #         + " + ".join(successors)
        #         + ";"
        #     )

        for action_idx, P_u in enumerate(transition_matrices):

            row = P_u[state]

            # Keep only significant probabilities
            kept = [
                (target, row[target])
                for target in range(n_states)
                if row[target] > tol
            ]

            # Renormalize after truncation
            total_prob = sum(p for _, p in kept)

            if total_prob <= 0:
                # numerical fallback
                successors = [f"1.0:(s'={state})"]
            else:
                successors = [
                    f"{p / total_prob:.16g}:(s'={target})"
                    for target, p in kept
                ]

            lines.append(
                f"    [u{action_idx}] s={state} -> "
                + " + ".join(successors)
                + ";"
            )

        lines.append("")

    lines.extend([
        "endmodule",
        "",
        'label "safe" = x_val >= safe_min & x_val <= safe_max;'
    ])

    return "\n".join(lines)





# Properties generation 
def generate_pctl(config) -> str:
    """Generates a PRISM pctl properties to verify.
    """


    if not config.is_controlled:

        return f"""
    const double T_max = {config.t_max};

    // 1. Path Safety Probability: 
    // Computes the probability that the Markov Chain STAYS continuously 
    // within the safe set [safe_min, safe_max] from time t=0 up to T_max.
    P=? [ G<=T_max "safe" ]

    // 2. Fixed-Horizon Target-Set Probability:
    // Computes the probability that the Markov Chain IS CURRENTLY in the 
    // safe set exactly at the final time T_max (Transient probability).
    P=? [ true U[T_max,T_max] "safe" ]

    """
    else:
        return f"""
    const int T_max = {int(config.t_max/config.dt)};

    // Find the strategy that MAXIMIZES the probability of staying safe
    Pmax=? [ G<=T_max "safe" ]


    """