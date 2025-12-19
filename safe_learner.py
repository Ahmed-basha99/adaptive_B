import torch
from cube import LocalCube
import configs as cfg

class SafeActiveLearner:
    def __init__(self):
        """
        Args:
            full_domain: Tensor (N, d) of all possible points.
            safe_threshold: Float h.
            params: Dict of hyperparameters (noise_std, delta_cube, etc.)
        """
        self.global_domain = cfg.DOMAIN
        self.safe_threshold = cfg.SAFE_THRESHOLD
        
        # global samples over all subdomains 
        self.x_sample = None
        self.y_sample = None
        
        # localization configs
        self.delta_cube = cfg.delta_cube
        self.num_local_cubes = cfg.num_local_cubes
        
        # track "interesting" Cubes (Set of tuples (i, k))
        # (-1, -1) => global domain
        self.interesting_domains = {(-1, -1)}

    def get_data(self):
        return self.x_sample, self.y_sample
    
    def add_data(self, x_new, y_new):
        y_new = y_new.flatten()
        if x_new.ndim == 1:
            x_new = x_new.unsqueeze(0)
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
        self.interesting_domains |= indices_set  
        self.interesting_domains.add((-1, -1))        

    def get_cube_data(self, i, k):
        """Helper to extract bounds and filter data for cube (i, k)."""
        if (i, k) == (-1, -1):
            
            lb = torch.min(self.global_domain, dim=0).values
            ub = torch.max(self.global_domain, dim=0).values
            return lb, ub, self.x_sample, self.y_sample, self.global_domain

        # local cube center 
        center = self.x_sample[i]
        radius = (k + 1) * self.delta_cube
        
        lb = center - radius
        ub = center + radius
        
        # clip global domain 
        lb = torch.max(lb, torch.min(self.global_domain, dim=0).values)
        ub = torch.min(ub, torch.max(self.global_domain, dim=0).values)
        
        # filter only samples that belongs to the specific cube
        in_bounds_data = torch.all((self.x_sample >= lb) & (self.x_sample <= ub), dim=1)
        local_x = self.x_sample[in_bounds_data]
        local_y = self.y_sample[in_bounds_data]
        
        # filter domain points that belong to the cube
        in_bounds_domain = torch.all((self.global_domain >= lb) & (self.global_domain <= ub), dim=1)
        local_domain = self.global_domain[in_bounds_domain]
        
        return lb, ub, local_x, local_y, local_domain

    def step(self): # ALGO 4 : aquestion 

        best_global_width = -1.0
        best_global_x = None
        best_cube_id = None

        current_cubes = list(self.interesting_domains)
        cubes = []
        flag = False
        for (i, k) in current_cubes:
            try:
                # local lower boundm, upper bound, local x and y samples, local domain discretization
                lb, ub, loc_x, loc_y, loc_dom = self.get_cube_data(i, k)
            except IndexError:
                continue
            
            if len(loc_dom) == 0: continue

            cube = LocalCube(
                loc_dom, lb, ub, loc_x, loc_y, (i,k))
            # call ALGO 3
            cube.compute_B() 
            cube.compute_bounds_and_safe_set()
            flag |= cube.unsafe_samples
            if cube.max_width > best_global_width:
                dists = torch.norm(self.x_sample - cube.best_x, dim=1)
                # print(torch.min(dists))
                # print(self.x_sample, cube.best_x)
                if torch.min(dists) > 1e-5:
                    best_global_width = cube.max_width
                    best_global_x = cube.best_x
                    best_cube_id = (i, k)
            # print(f'Cube (i={i}, k={k}) - Max Width: {cube.max_width:.4f}, sample: {cube.best_x}')
            cubes.append(cube)
            # If a cube has no safe points or uncertainty is low, remove from interesting set
            if cube.max_width <= 0: 
                self.interesting_domains.remove((i, k))
            
            # best_global_y = cfg.ground_truth_function(best_global_x) + 0.01 * torch.randn(1)
            # self.add_data(best_global_x, best_global_y.flatten())
            if (i,k) == (-1,-1) : 
                rmse = cube.evaluate_safe_rmse()
                print(f" Global Cube Safe RMSE: {rmse:.4f}")
        if flag : print(" Warning: Unsafe samples detected in safe set!")
        return best_global_x, best_cube_id, cubes 
    

    