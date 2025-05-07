"""Model implementation for TabSyn."""
# This file includes code adapted from the TabSyn project:
# https://github.com/amazon-science/tabsyn
# Licensed under the Apache License, Version 2.0
# Modifications have been made to suit this project's requirements.


import torch
import torch.nn as nn

from .diffusion_utils import EDMLoss


class PositionalEmbedding(nn.Module):
    """
    Positional embedding module.
    
    Args:
        num_channels (int): Number of channels
        max_positions (int): Maximum positions
        endpoint (bool): Whether to use endpoint
    """
    
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Positional embeddings
        """
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPDiffusion(nn.Module):
    """
    MLP-based diffusion model.
    
    Args:
        d_in (int): Input dimension
        dim_t (int): Time embedding dimension
    """
    
    def __init__(self, d_in, dim_t=512):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, noise_labels, class_labels=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            noise_labels (torch.Tensor): Noise level labels
            class_labels (torch.Tensor, optional): Class labels
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Get time embeddings
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        emb = self.time_embed(emb)
    
        # Project input and add time embeddings
        x = self.proj(x) + emb
        
        # Apply MLP
        return self.mlp(x)


class Precond(nn.Module):
    """
    Preconditioning module for diffusion model.
    
    Args:
        denoise_fn (nn.Module): Denoising network
        hid_dim (int): Hidden dimension
        sigma_min (float): Minimum supported noise level
        sigma_max (float): Maximum supported noise level
        sigma_data (float): Expected standard deviation of training data
    """
    
    def __init__(
        self,
        denoise_fn,
        hid_dim,
        sigma_min=0,
        sigma_max=float('inf'),
        sigma_data=0.5,
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            sigma (torch.Tensor): Noise levels
            
        Returns:
            torch.Tensor: Denoised tensor
        """
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        # Compute preconditioning parameters
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # Scale input
        x_in = c_in * x
        
        # Apply model
        F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

        # Apply skip connection and output scaling
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        
        return D_x

    def round_sigma(self, sigma):
        """Round sigma value."""
        return torch.as_tensor(sigma)


class Model(nn.Module):
    """
    Full diffusion model wrapper.
    
    Args:
        denoise_fn (nn.Module): Denoising network
        hid_dim (int): Hidden dimension
        P_mean (float): Mean of noise distribution
        P_std (float): Standard deviation of noise distribution
        sigma_data (float): Data standard deviation
        gamma (float): Scaling factor
        opts (dict, optional): Additional options
        pfgmpp (bool): Whether to use PFGM++ formulation
    """
    
    def __init__(
        self, 
        denoise_fn, 
        hid_dim, 
        P_mean=-1.2, 
        P_std=1.2, 
        sigma_data=0.5, 
        gamma=5, 
        opts=None, 
        pfgmpp=False
    ):
        super().__init__()

        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)

    def forward(self, x):
        """
        Forward pass (computes loss).
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Loss value
        """
        loss = self.loss_fn(self.denoise_fn_D, x)
        return loss.mean(-1).mean()