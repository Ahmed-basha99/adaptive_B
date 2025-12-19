import gpytorch
import torch

class GPRegressionModel(gpytorch.models.ExactGP):  
    """
    Exact GP Model copied from main.py.
    """
    def __init__(self, train_x, train_y, noise_std, lengthscale=0.1):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.tensor(noise_std**2)
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        self.kernel.lengthscale = lengthscale
        self.covar_module = self.kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
