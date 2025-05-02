"""Diffusion utilities for TabSyn."""

import torch
import numpy as np

# Diffusion parameters
SIGMA_MIN = 0.002
SIGMA_MAX = 80
RHO = 7
S_CHURN = 1
S_MIN = 0
S_MAX = float('inf')
S_NOISE = 1


def sample(net, num_samples, dim, num_steps=50, device='cuda:0'):
    """
    Sample from the diffusion model.
    
    Args:
        net: Denoising network
        num_samples: Number of samples to generate
        dim: Dimension of each sample
        num_steps: Number of sampling steps
        device: Computation device
        
    Returns:
        torch.Tensor: Generated samples
    """
    # Initialize random latents
    latents = torch.randn([num_samples, dim], device=device)

    # Create step indices
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    # Define sigma schedule
    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / RHO) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / RHO) - sigma_max ** (1 / RHO))) ** RHO
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Start sampling process
    x_next = latents.to(torch.float32) * t_steps[0]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next)

    return x_next


def sample_step(net, num_steps, i, t_cur, t_next, x_next):
    """
    Perform a single sampling step in the diffusion process.
    
    Args:
        net: Denoising network
        num_steps: Total number of steps
        i: Current step index
        t_cur: Current noise level
        t_next: Next noise level
        x_next: Current sample
        
    Returns:
        torch.Tensor: Updated sample
    """
    x_cur = x_next
    
    # Increase noise temporarily
    gamma = min(S_CHURN / num_steps, np.sqrt(2) - 1) if S_MIN <= t_cur <= S_MAX else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_NOISE * torch.randn_like(x_cur)
    
    # Euler step
    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction
    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


class EDMLoss:
    """
    Elucidating Diffusion Models (EDM) loss function.
    
    Implements the loss function from the paper:
    "Elucidating the Design Space of Diffusion-Based Generative Models"
    
    Args:
        P_mean (float): Mean of the log-normal noise distribution
        P_std (float): Standard deviation of the log-normal noise distribution
        sigma_data (float): Data standard deviation
        hid_dim (int): Hidden dimension
        gamma (float): Scaling factor
        opts (dict, optional): Additional options
    """
    
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim=100, gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts

    def __call__(self, denoise_fn, data):
        """
        Compute the EDM loss.
        
        Args:
            denoise_fn: Denoising network
            data: Input data
            
        Returns:
            torch.Tensor: Computed loss
        """
        # Sample noise levels
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        # Compute loss weights
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # Add noise to data
        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        
        # Denoise
        D_yn = denoise_fn(y + n, sigma)
    
        # Compute loss
        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)

        return loss