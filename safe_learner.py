import torch
import gpytorch
import numpy as np
from scipy.special import comb
import copy
from gp import GPRegressionModel
from cube import LocalCube
import configs as cfg
# --- 1. GP Model (Copied directly from main.py) ---


# --- 2. The Safe Active Learner Class ---
class SafeActiveLearner:
    def __init__(self, X_plot, safe_threshold, noise_std=0.01, lengthscale=0.1, 
                 delta_confidence=0.01, kappa_PAC=0.01, gamma_PAC=0.1, m_PAC=1000, alpha_bar=1.0):
        """
        Simplified class managing the Safe Active Learning loop.
        
        Args:
            X_plot: Tensor of shape (N, d) representing the discrete domain.
            safe_threshold: Float. The minimum safety value h.
            noise_std: Standard deviation of noise.
            delta_confidence: Confidence parameter for beta (delta).
            kappa_PAC: Risk tolerance for PAC bounds.
            gamma_PAC: Probability parameter for PAC bounds.
            m_PAC: Number of random scenarios.
            alpha_bar: Scaling factor for random weights.
        """
        # Configuration
        self.X_plot = X_plot
        self.safe_threshold = safe_threshold
        self.noise_std = noise_std
        self.lengthscale = lengthscale
        self.delta_confidence = delta_confidence
        self.kappa_PAC = kappa_PAC
        self.gamma_PAC = gamma_PAC
        self.m_PAC = m_PAC
        self.alpha_bar = alpha_bar

        # Data Storage
        self.x_sample = None
        self.y_sample = None
        
        # Internal State
        self.model = None
        self.K = None  # Covariance matrix
        self.B = 1.0   # Initial RKHS norm estimate
        self.beta = 2.0 
        self.mean = None
        self.var = None
        self.lcb = None
        self.ucb = None
        self.S = None  # Safe Set Mask

        # Domain bounds for sampling (Global approach)
        self.lb = torch.min(X_plot, dim=0).values
        self.ub = torch.max(X_plot, dim=0).values

    def get_data(self):
        return self.x_sample, self.y_sample
    
    def add_data(self, x_new, y_new):
        y_new = y_new.flatten()
            
        if self.x_sample is None:
            self.x_sample = x_new
            self.y_sample = y_new
        else:
            self.x_sample = torch.cat([self.x_sample, x_new], dim=0)

            if self.y_sample.ndim == 0:
                self.y_sample = self.y_sample.unsqueeze(0)
            if y_new.ndim == 0:
                y_new = y_new.unsqueeze(0)
            self.y_sample = torch.cat([self.y_sample, y_new], dim=0)
        
        dists = torch.max(torch.abs(self.x_sample - x_new), dim=1).values
        effect_tensor = dists.unsqueeze(1) <= torch.arange(1, self.num_local_cubes + 1) * self.delta_cube
        indices = torch.nonzero(effect_tensor, as_tuple=False)
        indices_set = {(i.item(), k.item()) for i, k in indices}
        self.interesting_domains |= indices_set  # set union
        self.interesting_domains.add((-1, -1))        

    def fit_model(self):  # not needed 
        """
        Recreates and fits the GP model.
        Corresponds to `compute_model` in safe_BO class.
        """
        # Re-initialize model with all current data
        self.model = GPRegressionModel(self.x_sample, self.y_sample, self.noise_std, self.lengthscale)
        # Compute covariance matrix K explicitly as done in main.py
        self.K = self.model(self.x_sample).covariance_matrix 
        self.model.eval() # Set to eval mode for predictions
        
    def estimate_pac_bound(self):
        """
        Estimates the RKHS norm B using the Scenario Approach.
        This is a direct adaptation of `compute_confidence_intervals_evaluation` (PAC block).
        """
        print(f'Getting PAC bounds now...')
        
        # 1. Determine Sampling Size N_hat
        N_hat = int(max(torch.round((torch.max(self.ub - self.lb)) * 500), len(self.y_sample) + 10))
        print (N_hat)
        x_interpol = self.x_sample
        y_interpol = self.y_sample
        list_random_RKHS_norms = []

        # 2. Scenario Loop
        for _ in range(self.m_PAC):
            # A. Generate Random Inputs X_c
            X_c = (torch.min(self.X_plot) - torch.max(self.X_plot)) * torch.rand(N_hat, self.x_sample.shape[1]) + torch.max(self.X_plot)
            
            # B. Enforce Head/Tail Structure
            X_c_tail = X_c[x_interpol.shape[0]:]
            X_c[:self.x_sample.shape[0]] = x_interpol
            
            # C. Randomize Tail Weights
            alpha_tail = -2 * self.alpha_bar * torch.rand(N_hat - len(y_interpol), 1) + self.alpha_bar
            
            # D. Interpolation Logic (Head Weights)
            # Matrix calculations to force the function to pass through observed data
            y_tail = self.model.kernel(x_interpol, X_c_tail).evaluate() @ alpha_tail
            y_head = (y_interpol - torch.squeeze(y_tail)).reshape(-1, 1)
            
            # Matrix inversion with nugget factor (1e-3)
            K_head_inv = torch.inverse(self.model.kernel(x_interpol, x_interpol).evaluate() + torch.eye(len(y_interpol)) * 1e-3)
            alpha_head = K_head_inv @ y_head
            
            # E. Compute Norm
            alpha = torch.cat((alpha_head, alpha_tail))
            random_RKHS_norm = torch.sqrt(alpha.T @ self.model.kernel(X_c, X_c).evaluate() @ alpha)
            list_random_RKHS_norms.append(random_RKHS_norm)

        # 3. Statistical Bound Selection (Binomial Test)
        numpy_list = [tensor.item() for tensor in list_random_RKHS_norms]
        numpy_list.sort()
        r_final = 0
        for r in range(self.m_PAC):
            summ = 0
            for i in range(r):
                summ += comb(self.m_PAC, i) * (self.gamma_PAC**i) * ((1 - self.gamma_PAC)**(self.m_PAC - i))
            
            # Check stopping condition: risk > kappa or bound exceeded
            if summ > self.kappa_PAC or self.B > numpy_list[-1-r]:
                break
            else:
                r_final = r
        
        # Update B (Algorithm 3)
        self.B = max(self.B, numpy_list[-1-r_final])
        print(f"Updated B: {self.B:.4f}")

    def compute_beta(self) : # double check which beta formula to use
        inside_log = torch.det(torch.eye(self.x_sample.shape[0]) + (1/self.noise_std * self.K))
        inside_sqrt = self.noise_std * torch.log(inside_log) - (2 * self.noise_std * torch.log(torch.tensor(self.delta_confidence)))
        self.beta = self.B + torch.sqrt(inside_sqrt) 
        return self.beta
    
    def get_posterior_mean_var(self):
        return self.mean, self.var
    
    def get_safe_set_and_bounds(self): 
        return self.S, self.lcb, self.ucb
    
    def compute_safe_set_and_bounds(self):
        self.model.eval()
        f_preds = self.model(self.X_plot)
        self.mean = f_preds.mean
        self.var = f_preds.variance

        # ALGO 3 : UPDATE norm estimation B
        self.estimate_pac_bound()

        self.compute_beta()
        
        self.lcb = self.mean - self.beta * torch.sqrt(self.var)
        self.ucb = self.mean + self.beta * torch.sqrt(self.var)

        self.S = self.lcb > self.safe_threshold

    def select_next_point(self): # ALGO 2
        """
        Active Learning Acquisition: Maximize uncertainty (width) within the Safe Set.
        Replaces 'maximizer_routine' and 'expander_routine' with pure uncertainty sampling.
        """

        self.compute_safe_set_and_bounds()  

        width = self.ucb - self.lcb
        
        # Filter for Safe Set
        if not torch.any(self.S):
            print("Warning: No safe points found. Using safe set from initialization if available.")
            # Fallback (should not happen if init is safe)
            return None, None

        # Mask unsafe widths with -infinity so they aren't picked
        safe_widths = width.clone()
        safe_widths[~self.S] = -float('inf')

        # Select index with maximum width
        best_idx = torch.argmax(safe_widths)
        x_next = self.X_plot[best_idx].unsqueeze(0)
        
        return x_next, best_idx

    def get_cube_data(self, i, k):
        """Helper to extract bounds and filter data for cube (i, k)."""
        if (i, k) == (-1, -1):
            # Global Cube logic
            lb = torch.min(self.X_global, dim=0).values
            ub = torch.max(self.X_global, dim=0).values
            return lb, ub, self.x_sample, self.y_sample, self.X_global

        # Local Cube Logic
        center = self.x_sample[i]
        radius = (k + 1) * self.delta_cube
        
        lb = center - radius
        ub = center + radius
        
        # Clip to global domain
        lb = torch.max(lb, torch.min(self.X_global, dim=0).values)
        ub = torch.min(ub, torch.max(self.X_global, dim=0).values)
        
        # Filter Data: "sample_indices = torch.all(..., axis=1)"
        in_bounds_data = torch.all((self.x_sample >= lb) & (self.x_sample <= ub), dim=1)
        local_x = self.x_sample[in_bounds_data]
        local_y = self.y_sample[in_bounds_data]
        
        # Filter Domain (Discretization)
        in_bounds_domain = torch.all((self.X_global >= lb) & (self.X_global <= ub), dim=1)
        local_domain = self.X_global[in_bounds_domain]
        
        return lb, ub, local_x, local_y, local_domain

    def step(self):

        best_global_width = -1.0
        best_global_x = None
        best_cube_id = None
        
        current_cubes = list(self.interesting_domains)
        print(f"Iterating through {len(current_cubes)} cubes...")
        
        for (i, k) in current_cubes:
            try:
                # local lower boundm, upper bound, local x and y samples, local domain discretization
                lb, ub, loc_x, loc_y, loc_dom = self._get_cube_data(i, k)
            except IndexError:
                # Handle edge case where i is invalid if we reset/cleared data
                continue
            
            if len(loc_dom) == 0: continue

            # 2. Create and Run Local Cube
            cube = LocalCube(
                loc_dom, lb, ub, loc_x, loc_y,
                self.safe_threshold, self.params['noise_std'], 
                self.params['delta_conf'], self.params
            )
            cube.run_estimation()
            
            # 3. Compare with Best
            # Logic: "max_uncertainty_interesting_local > max_uncertainty_interesting"
            if cube.max_width > best_global_width:
                # Check if this point is already sampled
                # "if not torch.any(torch.all(X_sample == x_new_current, axis=1))"
                # (Simple check: distance > 1e-6)
                dists = torch.norm(self.x_sample - cube.best_x, dim=1)
                if torch.min(dists) > 1e-5:
                    best_global_width = cube.max_width
                    best_global_x = cube.best_x
                    best_cube_id = (i, k)
            
            # 4. Pruning (Optional optimization from main.py)
            # If a cube has no safe points or uncertainty is low, remove from interesting set
            if cube.max_width <= 0:
                self.interesting_domains.remove((i, k))

        return best_global_x, best_cube_id