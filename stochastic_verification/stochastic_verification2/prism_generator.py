def generate_prism_model(config: SDEConfig, simulator) -> str:
    """Generates clean, parameterized PRISM model specification files.
    Generates an MDP instead of a CTMC when control space is specified.
    """
    h = config.h
    n_states = simulator.n_states
    x_min = config.x_safe_min - config.h
    safe_xmin = config.x_safe_min
    safe_xmax = config.x_safe_max
    init_state = simulator.start_idx

    model_type = "mdp" if config.is_controlled else "ctmc"
    
    lines = []
    lines.append(f"{model_type}\n")
    lines.append(f"const double h = {h};")
    lines.append(f"const int N = {n_states};")
    lines.append(f"const double x_min = {x_min};\n")
    lines.append(f"const double safe_min = {safe_xmin};")
    lines.append(f"const double safe_max = {safe_xmax};")
    lines.append(f"const int init_state = {init_state};\n")
    lines.append("formula x_val = x_min + s * h;\n")
    
    lines.append("module MCA_SDE")
    lines.append(f"    s : [0..N-1] init init_state;\n")
    
    # Generate transitions
    for i in range(n_states):
        x_val = x_min + i * h
        
        if config.is_controlled:
            # Generate non-deterministic choices for each control choice
            for u_idx, u_val in enumerate(config.U):
                mu_val = config.mu(x_val, u_val)
                sigma_val = config.sigma(x_val)
                
                # Compute up/down rate dynamics based on standard MCA framework
                rate_up = (0.5 * sigma_val**2 / (h**2)) + (max(mu_val, 0) / h)
                rate_down = (0.5 * sigma_val**2 / (h**2)) + (max(-mu_val, 0) / h)
                
                # Write command with action label [u0], [u1], etc.
                transitions = []
                if i < n_states - 1 and rate_up > 0:
                    transitions.append(f"{rate_up} : (s'={i+1})")
                if i > 0 and rate_down > 0:
                    transitions.append(f"{rate_down} : (s'={i-1})")
                
                if transitions:
                    lines.append(f"    [u{u_idx}] s={i} -> {' + '.join(transitions)};")
        else:
            # Standard uncontrolled CTMC transition rates code
            mu_val = config.mu(x_val)
            sigma_val = config.sigma(x_val)
            rate_up = (0.5 * sigma_val**2 / (h**2)) + (max(mu_val, 0) / h)
            rate_down = (0.5 * sigma_val**2 / (h**2)) + (max(-mu_val, 0) / h)
            
            transitions = []
            if i < n_states - 1 and rate_up > 0:
                transitions.append(f"{rate_up} : (s'={i+1})")
            if i > 0 and rate_down > 0:
                transitions.append(f"{rate_down} : (s'={i-1})")
                
            if transitions:
                lines.append(f"    s={i} -> {' + '.join(transitions)};")
                
    lines.append("\nendmodule\n")
    lines.append('label "safe" = x_val >= safe_min & x_val <= safe_max;')
    
    return "\n".join(lines)


def generate_pctl(config: SDEConfig) -> str:
    """Generates PRISM PCTL properties to verify.
    Uses Pmax bounded safety queries when control is specified.
    """
    T_max = config.t_max
    
    if config.is_controlled:
        return f'const double T_max = {T_max};\n\nPmax=? [ G<=T_max "safe" ]'
    else:
        return f'const double T_max = {T_max};\n\nP=? [ G<=T_max "safe" ]'