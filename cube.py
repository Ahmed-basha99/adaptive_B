import torch 
import numpy as np
import math
from gp import GPRegressionModel
import configs as cfg
from scipy.special import comb

class LocalCube:

    def __init__(self, domain_pts, lb, ub, local_x, local_y, cube_id):
        self.cube_id = cube_id
        self.local_domain = domain_pts # Points inside this cube only
        self.lb = lb
        self.ub = ub
        self.x_sample = local_x
        self.y_sample = local_y
        self.safe_threshold = cfg.SAFE_THRESHOLD
        self.noise_std = cfg.noise_std
        self.delta_confidence = cfg.delta_conf
        
        # PAC Parameters
        self.kappa_PAC = cfg.kappa
        self.gamma_PAC = cfg.gamma
        self.m_PAC = cfg.m
        self.alpha_bar = cfg.alpha

        # Outputs
        self.B = 0
        self.S = None
        self.ucb = None
        self.lcb = None
        self.beta = None
        self.max_width = 0.0
        self.best_x = None
        self.model = None
        self.unsafe_samples = False


    def compute_beta(self):
        # Fiedler et al. 2024 Equation (7); based on Abbasi-Yadkori 2013
        inside_log = torch.det(torch.eye(self.x_sample.shape[0]) + (1/self.noise_std*self.K))
        inside_sqrt = self.noise_std*torch.log(inside_log) - (2*self.noise_std*torch.log(torch.tensor(self.delta_confidence)))
        self.beta = self.B + torch.sqrt(inside_sqrt)
    
    def get_posterier_mean_var(self): 
        f_preds = self.model(self.local_domain)
        mean = f_preds.mean
        var = f_preds.variance
        return mean, var
    
    def compute_bounds_and_safe_set(self):
            # Compute (Q-Bounds)
            mean, var = self.get_posterier_mean_var()
            stdd = torch.sqrt(var)
            # Beta Calculation, and safe set update
            self.compute_beta()
            self.lcb = mean - self.beta * stdd
            self.ucb = mean + self.beta * stdd
            self.S = self.lcb > self.safe_threshold
            for i in range(len(self.S)) : 
                if self.S[i] and cfg.ground_truth_function(self.local_domain[i]) < self.safe_threshold : 
                    self.unsafe_samples = True

            # Find best candidate in this cube
            width = self.ucb - self.lcb
            if torch.any(self.S):
                safe_widths = width.clone()
                safe_widths[~self.S] = -float('inf')
                self.max_width = torch.max(safe_widths).item()
                best_idx = torch.argmax(safe_widths)
                self.best_x = self.local_domain[best_idx]
            else:
                self.max_width = -1.0
                self.best_x = None
            

    def compute_B(self): # ALGO 3
        print(f'Getting PAC bounds now...')

        self.model = GPRegressionModel(self.x_sample, self.y_sample, self.noise_std, lengthscale=0.1)
        self.K = self.model(self.x_sample).covariance_matrix
        self.model.eval()

        # 1. Determine Sampling Size N_hat
        # N_hat = int(max(torch.round((torch.max(self.ub - self.lb)) * 500), len(self.y_sample) + 10))
        N_hat = 500
        x_interpol = self.x_sample
        y_interpol = self.y_sample
        list_random_RKHS_norms = []

       
        if len(self.y_sample) > 0: 
            for _ in range(self.m_PAC):
                # A. Generate Random Inputs X_c
                X_c = (torch.min(self.local_domain) - torch.max(self.local_domain)) * torch.rand(N_hat, self.x_sample.shape[1]) + torch.max(self.local_domain)
                
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
                alpha_head = torch.inverse(self.model.kernel(x_interpol, x_interpol).evaluate()+torch.eye(len(y_interpol))*1e-3) @ y_head
                
                # E. Compute Norm
                alpha = torch.cat((alpha_head, alpha_tail))
                random_RKHS_norm = torch.sqrt(alpha.T @ self.model.kernel(X_c, X_c).evaluate() @ alpha)
                list_random_RKHS_norms.append(random_RKHS_norm)

        
        numpy_list = [tensor.item() for tensor in list_random_RKHS_norms]
        numpy_list.sort()
        r_final = 0
        for r in range(self.m_PAC):
            summ = 0
            for i in range(r):
                summ += comb(self.m_PAC, i) * (self.gamma_PAC**i) * ((1 - self.gamma_PAC)**(self.m_PAC - i))
            if summ > self.kappa_PAC or self.B > numpy_list[-1-r]:
                break
            else:
                r_final = r
        
        self.B = max(self.B, numpy_list[-1-r_final])
        print(f"Updated B: {self.B:.4f}")

    # make sure the plot method does not upate andything
    def plot(self, ax, cube_id) : 
        mean, _ = self.get_posterier_mean_var()
        S_mask = self.S.detach().numpy().flatten()
        Q_lower = self.lcb.detach().numpy()
        Q_upper = self.ucb.detach().numpy()
        trans = ax.get_xaxis_transform()
        safe_x = self.local_domain[S_mask]
        ax.scatter(
                safe_x,
                np.full_like(safe_x, 0.02),  
                marker='|',
                color='green',
                s=150,                     
                label='Safe Points',
                transform=trans           
            )
        ax.plot(self.x_sample.numpy(), self.y_sample.numpy(), 'ro', label='Training Data')
        ax.plot(self.local_domain.numpy(), cfg.ground_truth_function(self.local_domain).numpy(), 'bx--', label='Ground Truth')
        ax.plot(self.local_domain.numpy(), mean.detach().numpy(), 'k-', label='GP Mean')
        ax.plot(self.local_domain.numpy(), cfg.SAFE_THRESHOLD * torch.ones_like(self.local_domain).numpy(), 'g--', label='Safety Threshold')
        ax.fill_between(self.local_domain.numpy().flatten(), Q_lower, Q_upper, alpha=0.2, label='Q Bounds')
        # ax.axvline(self.lb.item(), color='purple', linestyle=':', 
        #            linewidth=2, alpha=0.7, label='Cube Bounds')
        # ax.axvline(self.ub.item(), color='purple', linestyle=':', 
        #            linewidth=2, alpha=0.7)
        if self.best_x is not None:
            ax.scatter(self.best_x.numpy(), 
                      cfg.ground_truth_function(self.best_x).numpy(),
                      c='orange', s=200, marker='*', 
                      edgecolors='black', linewidths=2,
                      label='Best Point', zorder=11)
        # cube id = 0
        if cube_id[0] == -1 :
            ax.set_title(f' Global Domain - Max W: {self.max_width:.4f} - B: {self.B:.4f}')
        else : ax.set_title(f' {cfg.DOMAIN[cube_id[0]]} - Max W: {self.max_width:.4f} - B: {self.B:.4f}')

    def evaluate_safe_rmse(self): 
        if not torch.any(self.S) : return float('inf')
        safe_points = self.local_domain > self.safe_threshold
        safe_x = self.local_domain[safe_points]
        true_y = cfg.ground_truth_function(safe_x)
        mean_preds = self.model(safe_x).mean
        rmse = torch.sqrt(torch.mean((true_y - mean_preds)**2)).item()
        return rmse