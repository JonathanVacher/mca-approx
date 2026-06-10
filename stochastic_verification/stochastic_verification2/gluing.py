# Inside your error bound evaluator (or main code executing Gluing derivatives):
def compute_total_error_bounds(gluing_obj, config: SDEConfig):
    """Computes error bounds for single or controlled SDE environments."""
    
    if not config.is_controlled:
        # Standard logic
        weak_error = gluing_obj.compute_weak_error()  # Uses mu(x)
        smoothing_error = gluing_obj.compute_smoothing_error()
        return weak_error, smoothing_error, weak_error + smoothing_error
    
    # If controlled, iterate over each possible value of u to find worst-case bounds
    max_weak_error = 0.0
    max_smoothing_error = 0.0
    
    # Temporarily substitute or evaluate mu(x, u) inside your symbolic/numerical grid
    for u_val in config.U:
        # Define a wrapped mu function fixing u to evaluate properties
        mu_fixed = lambda x: config.mu(x, u_val)
        
        # Perform your current derivative analysis / grid max extraction using mu_fixed
        # Example pseudo-calculation mapping onto your existing infrastructure:
        weak_err_u = gluing_obj.compute_weak_error_for_action(mu_fixed)
        smoothing_err_u = gluing_obj.compute_smoothing_error_for_action(mu_fixed)
        
        max_weak_error = max(max_weak_error, weak_err_u)
        max_smoothing_error = max(max_smoothing_error, max_smoothing_error_u)
        
    total_error = max_weak_error + max_smoothing_error
    return max_weak_error, max_smoothing_error, total_error