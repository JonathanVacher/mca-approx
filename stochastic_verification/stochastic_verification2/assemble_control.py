def _assemble_generator_controlled(self) -> list[np.ndarray]:
        """Assembles a list of continuous-time Markov Chain infinitesimal generator 
        matrices (Q), where each matrix corresponds to a discrete control choice u in U.
        
        Returns:
            list[np.ndarray]: A list of transition rate matrices of shape (n_states, n_states).
        """
        # Ensure that control actions are defined in the configuration
        if not hasattr(self.config, 'U') or self.config.U is None:
            raise ValueError("Control actions list 'U' must be specified in config to assemble controlled generators.")
            
        h = self.config.h
        q_matrices = []

        # Iterate through each available control input u
        for u_val in self.config.U:
            # Initialize an empty generator matrix for the current control action
            q_u = np.zeros((self.n_states, self.n_states))
            
            for i in range(self.n_states):
                x_val = self.x_space[i]
                
                # Evaluate drift and diffusion coefficients
                # mu now accepts two arguments: the state x_val and the control action u_val
                mu_val = self.config.mu(x_val, u_val)
                sigma_val = self.config.sigma(x_val)
                
                # Compute up and down transition rates based on the approximation framework
                rate_up = (0.5 * (sigma_val ** 2) / (h ** 2)) + (max(mu_val, 0) / h)
                rate_down = (0.5 * (sigma_val ** 2) / (h ** 2)) + (max(-mu_val, 0) / h)
                
                # Assign off-diagonal transition rates (ensuring boundary indices aren't exceeded)
                if i < self.n_states - 1:
                    q_u[i, i + 1] = rate_up
                if i > 0:
                    q_u[i, i - 1] = rate_down
                    
                # The diagonal entry must equal the negative sum of out-of-state rates 
                # so that each row sums to exactly 0 (conservation of probability)
                q_u[i, i] = -np.sum(q_u[i, :])
                
            q_matrices.append(q_u)
            
        return q_matrices