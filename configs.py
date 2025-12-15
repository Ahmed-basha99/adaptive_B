import torch 
import math 

# domain hyperparmeters
def ground_truth_1d(x):
    if x.ndim == 0: x = x.view(-1)
    return torch.sin(x * (6 * math.pi) + 0.5) 

ground_truth_function = ground_truth_1d
DOMAIN = torch.linspace(0, 1, 200).reshape(-1, 1)
SAFE_THRESHOLD = -0.2

INITIAL_SAFE_INDICES = torch.tensor([20])


# algo hyperparms  

m_PAC = 1000



# GP hyperparameters 

