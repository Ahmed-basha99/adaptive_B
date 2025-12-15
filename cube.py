import torch 
import numpy
import math
from gp import GPRegressionModel
import configs as cfg

class LocalCube:
    """
    Performs the PAC estimation and Safe Set calculation for a local region.
    """
    def __init__(self, domain_pts, lb, ub, local_x, local_y, 
                 safe_threshold, noise_std, delta_conf, params):
        self.X_plot = domain_pts # Points inside this cube only
        self.lb = lb
        self.ub = ub
        self.x_sample = local_x
        self.y_sample = local_y
        self.safe_threshold = safe_threshold
        self.noise_std = noise_std
        self.delta_confidence = delta_conf
        
        # PAC Parameters
        self.kappa_PAC = params['kappa']
        self.gamma_PAC = params['gamma']
        self.m_PAC = params['m']
        self.alpha_bar = params['alpha']
        
        # Outputs
        self.B = 1.0 # Default starting B
        self.S = None
        self.ucb = None
        self.lcb = None
        self.beta = None
        self.max_width = 0.0
        self.best_x = None

    def run_estimation(self):
        """Runs the complete GP -> PAC -> SafeSet pipeline for this cube."""
        if len(self.x_sample) == 0: return

        # A. Fit Model
        model = GPRegressionModel(self.x_sample, self.y_sample, self.noise_std, lengthscale=0.1)
        K = model(self.x_sample).covariance_matrix
        model.eval()

        # B. Estimate PAC Bound (Algorithm 3)
        # (Simplified loop for brevity; logic identical to previous code)
        N_hat = int(max(torch.round((torch.max(self.ub - self.lb)) * 500), len(self.y_sample) + 10))
        list_norms = []
        
        # Only run PAC if we have enough data to be meaningful
        if len(self.y_sample) > 0:
            for _ in range(self.m_PAC):
                # Generate random points inside GLOBAL bounds (approximation for simplicity)
                # or local bounds. main.py uses discr_domain for some parts but random logic is robust.
                # Here we stick to the scenario logic:
                x_interpol = self.x_sample
                y_interpol = self.y_sample
                
                # Random scenario generation
                # Note: Sampling randomly from [lb, ub] for constructing B
                X_c = (self.lb - self.ub) * torch.rand(N_hat, self.x_sample.shape[1]) + self.ub
                X_c[:len(x_interpol)] = x_interpol
                
                # Tail weights
                alpha_tail = -2 * self.alpha_bar * torch.rand(N_hat - len(y_interpol), 1) + self.alpha_bar
                y_tail = model.kernel(x_interpol, X_c[len(x_interpol):]).evaluate() @ alpha_tail
                
                # Head weights
                K_inv = torch.inverse(model.kernel(x_interpol, x_interpol).evaluate() + torch.eye(len(y_interpol))*1e-3)
                y_head = (y_interpol - torch.squeeze(y_tail)).reshape(-1, 1)
                alpha_head = K_inv @ y_head
                
                alpha = torch.cat((alpha_head, alpha_tail))
                norm = torch.sqrt(alpha.T @ model.kernel(X_c, X_c).evaluate() @ alpha)
                list_norms.append(norm)

            # Statistical Selection
            list_norms = [t.item() for t in list_norms]
            list_norms.sort()
            r_final = 0
            for r in range(self.m_PAC):
                summ = sum([comb(self.m_PAC, i)*(self.gamma_PAC**i)*((1-self.gamma_PAC)**(self.m_PAC-i)) for i in range(r)])
                if summ > self.kappa_PAC: break
                r_final = r
            self.B = max(self.B, list_norms[-1-r_final])

        # C. Compute Safe Set (Q-Bounds)
        f_preds = model(self.X_plot)
        mean = f_preds.mean
        var = f_preds.variance
        
        # Beta Calculation
        inside_log = torch.det(torch.eye(len(self.x_sample)) + (1/self.noise_std * K))
        self.beta = self.B + torch.sqrt(self.noise_std * torch.log(inside_log) - (2 * self.noise_std * torch.log(torch.tensor(self.delta_confidence))))
        
        self.lcb = mean - self.beta * torch.sqrt(var)
        self.ucb = mean + self.beta * torch.sqrt(var)
        self.S = self.lcb > self.safe_threshold

        # D. Find best candidate in this cube
        width = self.ucb - self.lcb
        if torch.any(self.S):
            safe_widths = width.clone()
            safe_widths[~self.S] = -float('inf')
            self.max_width = torch.max(safe_widths).item()
            best_idx = torch.argmax(safe_widths)
            self.best_x = self.X_plot[best_idx]
        else:
            self.max_width = -1.0
            self.best_x = None