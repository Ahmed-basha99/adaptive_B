import torch 
import math 

# domain hyperparmeters
def ground_truth_1d(x):
    if x.ndim == 0: x = x.view(-1)
    return torch.sin(x * (6 * math.pi) + 0.5) 

ground_truth_function = ground_truth_1d
DOMAIN = torch.linspace(0, 1, 200).reshape(-1, 1)
SAFE_THRESHOLD = -0.2

INITIAL_SAFE_INDICES = torch.tensor([20,140, 80, 199])

# algo hyperparms  
noise_std  = 0.05
delta_cube = 0.1
num_local_cubes = 0
delta_conf = 0.01
kappa = 0.005
gamma = 0.1
alpha = 0.5
m = 200



# GP hyperparameters 

