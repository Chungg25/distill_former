import torch
from torch import nn

class DEMA(nn.Module):
    """
    Double Exponential Moving Average (DEMA) block to highlight the trend of time series
    """
    def __init__(self, alpha, beta):
        super(DEMA, self).__init__()
        # self.alpha = nn.Parameter(alpha)    # Learnable alpha
        # self.beta = nn.Parameter(beta)      # Learnable beta
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        # x: [Batch, Time, Channel]
        B, T, C = x.shape
        s = torch.zeros((B, T, C), device=x.device, dtype=x.dtype)
        b = torch.zeros((B, T, C), device=x.device, dtype=x.dtype)
        s[:,0,:] = x[:,0,:]
        b[:,0,:] = x[:,1,:] - x[:,0,:]
        for t in range(1, T):
            s[:,t,:] = self.alpha * x[:,t,:] + (1-self.alpha) * (s[:,t-1,:] + b[:,t-1,:])
            b[:,t,:] = self.beta * (s[:,t,:] - s[:,t-1,:]) + (1-self.beta) * b[:,t-1,:]
        return s