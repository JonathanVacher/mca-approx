"""Generates clean, parameterized PRISM CTMC model specification files."""


def generate_prism_model(
    tau: float,
    sig: float,
    h: float,
    n_states: int,
    x_min: float,
    safe_xmin: float,
    safe_xmax: float,
    init_state: int = 20,
) -> str:
    """Generates a PRISM Continuous-Time Markov Chain (CTMC) model string

    based on a nodal generator discretization matrix.
    """
    return f"""ctmc

// SDE parameters
const double tau = {tau};
const double sig = {sig};

// Grid and State Space parameters
const double h = {h};
const int N = {n_states};
const double x_min = {x_min};

// Safety Set boundaries
const double safe_min = {safe_xmin};
const double safe_max = {safe_xmax};

// The physical state space value 'x' for a given index 's'
formula x_val = x_min + s * h;

// Transition rates based on the nodal CTMC generator discretization
formula rate_up   = (0.5 * sig * sig / (h * h)) - (x_val / (tau * 2.0 * h));
formula rate_down = (0.5 * sig * sig / (h * h)) + (x_val / (tau * 2.0 * h));

module MCA_Ornstein_Uhlenbeck

    // State index
    s : [0..N-1] init {init_state};

    // Internal state transitions
    [] s > 0 & s < N-1 -> rate_up   : (s'=s+1) + rate_down : (s'=s-1);
    [] s = 0          -> rate_up   : (s'=0) + rate_down : (s'=0);
    [] s = N-1        -> rate_down : (s'=N-1) + rate_up   : (s'=N-1);

endmodule

// Labeling safety criteria
label "safe" = x_val >= safe_min & x_val <= safe_max;
"""





def generate_pctl(
    T_max: float
) -> str:
    """Generates a PRISM pctl properties to verify.
    """
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