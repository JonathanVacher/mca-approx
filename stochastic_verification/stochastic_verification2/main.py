# Cell 1: Controlled Definition Setup
tau = 0.5 
sigma = 1.5

# Controlled SDE: mu receives state 'x' and control action 'u'
controlled_mu = lambda x, u: -x/tau + u

config = SDEConfig(
    mu=controlled_mu,
    sigma=lambda x: sigma,
    dt=0.01, 
    t_max=0.5, 
    h=0.001, 
    x_safe_min=-1.0,
    x_safe_max=1.0,
    epsilon=0.3,
    x_min=-1.5, 
    x_max=1.5,
    x_init=-0.5,
    U=[-1.0, 1.0] # Specifying control values triggers MDP and controlled variants
)

# Cell 2: Parse Results Logic Adjustments
# When reading the output string file ('res.txt') produced by PRISM execution:
with open("res.txt", "r") as f:
    prism_output = f.read()

# Updated Regex to find either single P result or non-deterministic MDP Pmax result
match = re.search(r"Result:\s*([0-9.]+)", prism_output)
if match:
    safe_prob = float(match.group(1))
    print(f"PRISM Verification Safety Probability: {safe_prob:.6f}")
    
    # Compute error bounds interval according to your worst-case compiled bounds
    safe_l = max(min(safe_prob - total_error, 1.0), 0.0)
    safe_r = max(min(safe_prob + total_error, 1.0), 0.0)
    print(f"SDE Ensured Safety Bound Interval: [{safe_l:.6f}, {safe_r:.6f}]")