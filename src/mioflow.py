"""
TODO:
1. per-time-point loss weights
2. deprecate local training
3. add weight initialization
4. test on gpu
5. test different activation functions
""" 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import ot
from torchdiffeq import odeint
import torchsde
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple

class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, momentum_beta=0.0):
        super().__init__()
        self.momentum_beta = momentum_beta
        self.previous_v = None
        
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), # additional dim for time t.
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Kaiming initialization for SiLU (better than default for ReLU-like activations)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def reset_momentum(self):
        """Reset momentum state before integration"""
        self.previous_v = None

    def forward(self, t, x):
        # t is scalar, x is [batch_size, input_dim]
        # Expand t to [batch_size, 1] to match x's batch dimension
        t_expanded = t.expand(x.size(0), 1)
        input = torch.cat([t_expanded, x], dim=-1)
        dxdt = self.model(input)
        
        # Apply momentum if enabled
        if self.momentum_beta > 0.0:
            if self.previous_v is None or self.previous_v.shape[0] != x.shape[0]:
                self.previous_v = torch.zeros_like(dxdt)
            dxdt = self.momentum_beta * self.previous_v + (1 - self.momentum_beta) * dxdt
            self.previous_v = dxdt.detach()
        
        return dxdt


class SDEFunc(nn.Module):
    """
    Neural SDE with diagonal noise.
    Implements both drift (f) and diffusion (g) terms.
    """
    noise_type = 'diagonal'
    sde_type = 'ito'
    
    def __init__(self, input_dim, hidden_dim, diffusion_scale=0.1, diffusion_init_scale=0.1, momentum_beta=0.0):
        super().__init__()
        self.diffusion_scale = diffusion_scale
        self.diffusion_init_scale = diffusion_init_scale
        self.momentum_beta = momentum_beta
        self.previous_v = None
        
        # Drift network (same complexity as ODEFunc)
        self.drift_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Diffusion network (simpler: 1 hidden layer)
        # Softplus ensures positive diffusion: g ∈ [0, ∞)
        self.diffusion_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softplus(),
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Kaiming initialization for both drift and diffusion networks
        for net in [self.drift_net, self.diffusion_net]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Scale down diffusion network for stability
        # diffusion_init_scale controls initial noise level
        # 0.1 = small noise, 0.0 = pure ODE initially, 1.0 = no scaling
        if self.diffusion_init_scale != 1.0:
            with torch.no_grad():
                for m in self.diffusion_net.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.data *= self.diffusion_init_scale
    
    def reset_momentum(self):
        """Reset momentum state before integration"""
        self.previous_v = None
    
    def f(self, t, x):
        """Drift term"""
        t_expanded = t.expand(x.size(0), 1)
        input = torch.cat([t_expanded, x], dim=-1)
        drift = self.drift_net(input)
        
        # Apply momentum if enabled
        if self.momentum_beta > 0.0:
            if self.previous_v is None or self.previous_v.shape[0] != x.shape[0]:
                self.previous_v = torch.zeros_like(drift)
            drift = self.momentum_beta * self.previous_v + (1 - self.momentum_beta) * drift
            self.previous_v = drift.detach()
        
        return drift
    
    def g(self, t, x):
        """Diffusion term - scaled by diffusion_scale"""
        t_expanded = t.expand(x.size(0), 1)
        input = torch.cat([t_expanded, x], dim=-1)
        return self.diffusion_scale * self.diffusion_net(input)

class GrowthRateModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, use_time=True):
        super().__init__()
        self.use_time = use_time
        # Add 1 to input_dim if we are using time
        actual_input_dim = input_dim + 1 if use_time else input_dim
        
        self.net = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.softplus = nn.Softplus()
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize last layer bias to 0.5413 so initial mass is approx 1.0 (Softplus(0.5413) ~ 1.0)
        # and weights to 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.5413)

    def forward(self, x, t=None):
        if self.use_time:
            if t is None:
                raise ValueError("t must be provided if use_time is True")
            # Expand scalar t or vector t to match x batch size
            if t.dim() == 0:
                t_expanded = t.expand(x.size(0), 1)
            elif t.dim() == 1 and t.size(0) == 1:
                t_expanded = t.expand(x.size(0), 1)
            else:
                t_expanded = t.view(-1, 1)
            input_feats = torch.cat([x, t_expanded], dim=-1)
        else:
            input_feats = x
            
        # Mass = Softplus(net(input_feats)) + 1e-4 to ensure strict positivity
        out = self.net(input_feats)
        return self.softplus(out).squeeze(-1) + 1e-4


def ot_loss(source, target, source_mass=None, target_mass=None, reg=0.1):
    use_sinkhorn = source_mass is not None or target_mass is not None
    
    if use_sinkhorn:
        # Mask out explicitly dead points so they don't get forced into the balanced EMD matching
        mask_threshold = 0.05
        mask = source_mass > mask_threshold
        
        if not mask.any():
            # If all points in the batch are dead, no OT cost to transport them
            return torch.tensor(0.0, device=source.device, requires_grad=True)
            
        active_source = source[mask]
        active_mass = source_mass[mask]
        M = torch.cdist(active_source, target)**2

        # Standardize mass to sum to 1.0 for balanced OT
        mu = active_mass / active_mass.sum()
        nu = torch.tensor(ot.unif(target.size(0)), dtype=target.dtype, device=target.device)
        
        # We detach the mass inputs so the ODE solver only sees the spatial vector field gradients
        loss = ot.emd2(mu.detach(), nu.detach(), M)
        return loss
    else:
        M = torch.cdist(source, target)**2
        # Standard balanced uniform EMD for regular MIOFlow
        mu = torch.tensor(ot.unif(source.size(0)), dtype=source.dtype, device=source.device)
        nu = torch.tensor(ot.unif(target.size(0)), dtype=target.dtype, device=target.device)
        return ot.emd2(mu, nu, M)

def energy_loss(model, x0, t_seq, is_sde=False, dt=0.1, lambda_f=1.0, lambda_g=0.0):
    """
    Compute energy loss by evaluating vector field magnitude along the trajectory.
    For ODE: penalizes ||f||^2
    For SDE: penalizes lambda_f * ||f||^2 + lambda_g * ||g||^2
    
    Recommended settings:
    - lambda_f=1.0, lambda_g=0.0: Only penalize drift (diffusion controlled by scale)
    - lambda_f=1.0, lambda_g=1.0: Penalize both equally
    - lambda_f=1.0, lambda_g=2.0: Penalize diffusion more (keep noise small)

    Args:
        model: ODEFunc or SDEFunc model
        x0: Initial points [batch_size, input_dim]
        t_seq: Time sequence [num_times]
        is_sde: Whether model is SDE (has f and g methods)
        dt: Time step for SDE integration
        lambda_f: Weight for drift penalty (default 1.0)
        lambda_g: Weight for diffusion penalty (default 0.0 - don't penalize)

    Returns:
        Energy loss (mean squared magnitude of vector field along trajectory)
    """
    # Reset momentum before integration
    model.reset_momentum()
    
    # Compute the full trajectory
    if is_sde:
        trajectory = torchsde.sdeint_adjoint(model, x0, t_seq, dt=dt, method='euler')
    else:
        trajectory = odeint(model, x0, t_seq)

    total_energy = 0.0
    num_evaluations = 0

    # Evaluate vector field at each point along the trajectory
    for i, t_val in enumerate(t_seq):
        x_t = trajectory[i]  # [batch_size, input_dim]
        t_tensor = t_val.clone().detach() if torch.is_tensor(t_val) else torch.tensor(t_val, device=x_t.device, dtype=x_t.dtype)

        if is_sde:
            # For SDE: separate penalties for drift and diffusion
            f_val = model.f(t_tensor, x_t)
            g_val = model.g(t_tensor, x_t)
            total_energy += lambda_f * torch.sum(f_val ** 2) + lambda_g * torch.sum(g_val ** 2)
        else:
            # For ODE: energy from vector field
            dx_dt = model(t_tensor, x_t)
            total_energy += torch.sum(dx_dt ** 2)
        
        num_evaluations += x_t.size(0)

    return total_energy / num_evaluations

def density_loss(source, target, source_mass=None, top_k=5, hinge_value=0.01):
    """
    Density loss that encourages points to be close to target distribution.
    Uses hinge loss on k-nearest neighbor distances.
    Masks out points that have mass <= 0.05.
    """
    if source_mass is not None:
        mask = source_mass > 0.05
        if not mask.any():
            return torch.tensor(0.0, device=source.device, requires_grad=True)
        active_source = source[mask]
    else:
        active_source = source
        
    c_dist = torch.cdist(active_source, target)
    values, _ = torch.topk(c_dist, top_k, dim=1, largest=False, sorted=False)
    values = torch.clamp(values - hinge_value, min=0.0)
    return torch.mean(values)


def infer(x0, model, t_seq, dt=0.1):
    """
    Run inference with ODE or SDE model.
    
    Args:
        x0: Initial condition [batch_size, input_dim]
        model: ODEFunc or SDEFunc
        t_seq: Time sequence tensor
        dt: Time step for SDE integration (ignored for ODE)
    
    Returns:
        Trajectory [time_steps, batch_size, input_dim]
    """
    # Reset momentum before integration
    model.reset_momentum()
    
    is_sde = hasattr(model, 'f') and hasattr(model, 'g')
    if is_sde:
        return torchsde.sdeint_adjoint(model, x0, t_seq, dt=dt, method='euler')
    else:
        return odeint(model, x0, t_seq)


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series data with variable number of points per time step.
    Data format: list of (X_t, t) tuples, where X_t has shape [n_points, dim]
    """
    def __init__(self, time_series_data: List[Tuple[np.ndarray, float]]):
        """
        Args:
            time_series_data: List of (X_t, t) tuples, where X_t is [n_points, dim] array
        """
        self.time_series_data = time_series_data
        self.times = [t for _, t in time_series_data]

    def __len__(self):
        return len(self.time_series_data) - 1  # Number of intervals

    def __getitem__(self, idx):
        """
        Returns data for training interval idx -> idx+1
        """
        X_t, t_start = self.time_series_data[idx]
        X_t1, t_end = self.time_series_data[idx + 1]

        return {
            'X_start': torch.tensor(X_t, dtype=torch.float32),
            'X_end': torch.tensor(X_t1, dtype=torch.float32),
            't_start': t_start,
            't_end': t_end,
            'interval_idx': idx
        }

    def get_time_sequence(self, start_idx=0, end_idx=None):
        """Get time sequence from start_idx to end_idx"""
        if end_idx is None:
            end_idx = len(self.times)
        return torch.tensor(self.times[start_idx:end_idx], dtype=torch.float32)

    def get_initial_condition(self, start_idx=0):
        """Get initial condition X_0"""
        X_0, _ = self.time_series_data[start_idx]
        return torch.tensor(X_0, dtype=torch.float32)


def compute_uot_growth_rates(dataset: TimeSeriesDataset, reg_m=[1.0, 100.0], div='l2'):
    """
    Computes Unbalanced Optimal Transport (UOT) mass for each point in the dataset 
    by solving sequential UOT problems between timepoints.
    
    A lower reg_m (e.g., 1.0) allows MASSIVE destruction of points, creating huge variance.
    
    Returns a list of 1D tensors, where each tensor contains the target mass 
    for the points at the corresponding time index.
    """
    growth_rates = []
    
    for i in range(len(dataset)):
        batch = dataset[i]
        x0 = batch['X_start'].cpu().numpy()
        x1 = batch['X_end'].cpu().numpy()
        
        m, n = ot.unif(len(x0)), ot.unif(len(x1))
        M = ot.dist(x0, x1)
        plan = ot.unbalanced.mm_unbalanced(m, n, M, reg_m, div=div)
        
        gr = plan.sum(axis=1) * plan.shape[0]
        growth_rates.append(torch.tensor(gr, dtype=torch.float32))
        
    return growth_rates

def pretrain_growth_model(
    model: GrowthRateModel, 
    dataset: TimeSeriesDataset, 
    uot_masses: List[torch.Tensor],
    num_epochs: int = 50, 
    learning_rate: float = 1e-3, 
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    scheduler_type: str = None,  # 'step', 'exponential', 'cosine', or None
    scheduler_step_size: int = 30,  # For StepLR
    scheduler_gamma: float = 0.5,  # Decay factor for schedulers
    scheduler_t_max: int = None,  # For CosineAnnealingLR, defaults to num_epochs
    scheduler_min_lr: float = 0.0   # Minimum learning rate for cosine scheduler
):
    """
    Pretrains the GrowthRateModel using MSE loss to match the UOT computed masses.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create learning rate scheduler if specified
    scheduler = None
    if scheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    elif scheduler_type == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    elif scheduler_type == 'cosine':
        t_max = scheduler_t_max if scheduler_t_max is not None else num_epochs
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=scheduler_min_lr)

    criterion = nn.MSELoss()
    
    model.train()
    
    all_x = []
    all_t = []
    all_target_mass = []
    for i in range(len(dataset)):
        batch = dataset[i]
        all_x.append(batch['X_start'])
        t_start = batch['t_start']
        all_t.append(torch.full((batch['X_start'].size(0), 1), t_start, dtype=torch.float32))
        all_target_mass.append(uot_masses[i])
        
        # Explicitly anchor the final timepoint!
        if i == len(dataset) - 1:
            all_x.append(batch['X_end'])
            t_end = batch['t_end']
            all_t.append(torch.full((batch['X_end'].size(0), 1), t_end, dtype=torch.float32))
            all_target_mass.append(torch.ones(batch['X_end'].size(0), dtype=torch.float32))
        
    all_x = torch.cat(all_x, dim=0).to(device)
    all_t = torch.cat(all_t, dim=0).to(device)
    all_target_mass = torch.cat(all_target_mass, dim=0).to(device)
    
    pbar = tqdm(range(num_epochs), desc='Pre-training Growth Model')
    for epoch in pbar:
        optimizer.zero_grad()
        
        if getattr(model, 'use_time', False):
            predicted_mass = model(all_x, all_t)
        else:
            predicted_mass = model(all_x)
            
        loss = criterion(predicted_mass, all_target_mass)
        
        loss.backward()
        optimizer.step()
        
        # Step the scheduler if specified
        if scheduler is not None:
            scheduler.step()
        
        postfix_dict = {'MSE': f'{loss.item():.4f}'}
        if scheduler is not None:
            postfix_dict['LR'] = f'{optimizer.param_groups[0]["lr"]:.2e}'
            
        pbar.set_postfix(postfix_dict)


def train_mioflow(
    model,
    dataset: TimeSeriesDataset,
    num_epochs: int,
    batch_size: int = None,  # Number of points to sample per time step (None = use all)
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lambda_ot: float = 1.0,
    lambda_density: float = 0.1,
    lambda_energy: float = 0.01,
    lambda_energy_f: float = 1.0,  # Weight for drift energy (SDE only)
    lambda_energy_g: float = 0.0,  # Weight for diffusion energy (SDE only)
    energy_time_steps: int = 10,
    sde_dt: float = 0.1,  # Time step for SDE integration
    grad_clip: float = 1.0,  # Gradient clipping for SDE (None = no clipping)
    scheduler_type: str = None,  # 'step', 'exponential', 'cosine', or None
    scheduler_step_size: int = 30,  # For StepLR
    scheduler_gamma: float = 0.5,  # Decay factor for schedulers
    scheduler_t_max: int = None,  # For CosineAnnealingLR, defaults to num_epochs
    scheduler_min_lr: float = 0.0,  # Minimum learning rate for cosine scheduler
    growth_rate_model = None,  # Growth rate model
    growth_rate_lr: float = 1e-4  # Learning rate for growth rate model
) -> Dict:
    """
    Train MIOFlow model with ODE or SDE.

    Args:
        model: ODEFunc or SDEFunc model
        dataset: TimeSeriesDataset
        num_epochs: Number of training epochs
        batch_size: Number of points to sample per time step (None = use all)
        learning_rate: Learning rate
        device: Device to train on
        lambda_ot: Weight for OT loss
        lambda_density: Weight for density loss
        lambda_energy: Weight for energy regularization
        lambda_energy_f: Weight for drift energy (SDE only, default 1.0)
        lambda_energy_g: Weight for diffusion energy (SDE only, default 0.0)
        energy_time_steps: Number of time steps for energy evaluation
        sde_dt: Time step for SDE integration (ignored for ODE)
        grad_clip: Max gradient norm for clipping (SDE only, None = no clipping, default 1.0)
        scheduler_type: Learning rate scheduler type ('step', 'exponential', 'cosine', or None)
        scheduler_step_size: Step size for StepLR scheduler
        scheduler_gamma: Decay factor for schedulers
        scheduler_t_max: Maximum number of iterations for CosineAnnealingLR (defaults to num_epochs)
        scheduler_min_lr: Minimum learning rate for cosine scheduler (default 0.0)
        growth_rate_model: Optional GrowthRateModel instance
        growth_rate_lr: Learning rate for growth_rate_model parameters

    Returns:
        Training history
        
    Note:
        For SDE, energy = lambda_energy * (lambda_energy_f * ||f||^2 + lambda_energy_g * ||g||^2)
        Recommended: lambda_energy_f=1.0, lambda_energy_g=0.0 (only penalize drift)
    """
    model = model.to(device)
    
    if growth_rate_model is not None:
        growth_rate_model = growth_rate_model.to(device)
        optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': learning_rate},
            {'params': growth_rate_model.parameters(), 'lr': growth_rate_lr}
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    # Detect if model is SDE
    is_sde = hasattr(model, 'f') and hasattr(model, 'g')

    # Create learning rate scheduler if specified
    scheduler = None
    if scheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    elif scheduler_type == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    elif scheduler_type == 'cosine':
        t_max = scheduler_t_max if scheduler_t_max is not None else num_epochs
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=scheduler_min_lr)

    history = {
        'epoch': [],
        'total_loss': [],
        'ot_loss': [],
        'density_loss': [],
        'energy_loss': []
    }

    epoch_pbar = tqdm(range(num_epochs), desc='Epochs')
    for epoch in epoch_pbar:
        epoch_losses = {'total': 0.0, 'ot': 0.0, 'density': 0.0, 'energy': 0.0}
        num_batches = 0

        # Train each time interval separately
        for interval_idx in range(len(dataset)):
            batch = dataset[interval_idx]

            X_start = batch['X_start'].to(device)
            X_end = batch['X_end'].to(device)
            t_start = batch['t_start']
            t_end = batch['t_end']

            # Sample points if batch_size specified
            if batch_size is not None:
                min_size = min(X_start.size(0), X_end.size(0))
                effective_batch_size = min(batch_size, min_size)
                indices = torch.randperm(min_size)[:effective_batch_size]
                X_start = X_start[indices]
                X_end = X_end[indices]

            # Integrate from X_start to predict X_end
            model.reset_momentum()
            t_interval = torch.tensor([t_start, t_end], device=device, dtype=torch.float32)
            if is_sde:
                X_pred = torchsde.sdeint_adjoint(model, X_start, t_interval, dt=sde_dt, method='euler')[1]
            else:
                X_pred = odeint(model, X_start, t_interval)[1]

            # Compute losses (only when weights are non-zero)
            total_loss = 0.0
            ot_loss_val = torch.tensor(0.0, device=device)
            density_loss_val = torch.tensor(0.0, device=device)
            energy_loss_val = torch.tensor(0.0, device=device)

            source_mass = None
            if growth_rate_model is not None:
                if getattr(growth_rate_model, 'use_time', False):
                    t_tensor = torch.tensor([t_start], device=device, dtype=torch.float32)
                    source_mass = growth_rate_model(X_start, t_tensor)
                else:
                    source_mass = growth_rate_model(X_start)

            if lambda_ot > 0:
                ot_loss_val = ot_loss(X_pred, X_end, source_mass=source_mass)
                total_loss += lambda_ot * ot_loss_val
                    
            if lambda_density > 0:
                density_loss_val = density_loss(X_pred, X_end, source_mass=source_mass)
                total_loss += lambda_density * density_loss_val

            if lambda_energy > 0:
                # Create denser time grid for energy loss
                energy_t_seq = torch.linspace(t_start, t_end, energy_time_steps, device=device, dtype=torch.float32)
                energy_loss_val = energy_loss(model, X_start, energy_t_seq, is_sde=is_sde, dt=sde_dt, 
                                             lambda_f=lambda_energy_f, lambda_g=lambda_energy_g)
                total_loss += lambda_energy * energy_loss_val

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients to prevent explosion (especially important for SDE)
            if is_sde and grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            optimizer.step()

            # Accumulate losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['ot'] += ot_loss_val.item()
            epoch_losses['density'] += density_loss_val.item()
            epoch_losses['energy'] += energy_loss_val.item()
            num_batches += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # Step the scheduler if specified
        if scheduler is not None:
            scheduler.step()

        # Record history
        history['epoch'].append(epoch + 1)
        history['total_loss'].append(epoch_losses['total'])
        history['ot_loss'].append(epoch_losses['ot'])
        history['density_loss'].append(epoch_losses['density'])
        history['energy_loss'].append(epoch_losses['energy'])

        # Update progress bar with loss information
        postfix_dict = {
            'Total': f'{epoch_losses["total"]:.4f}',
            'OT': f'{epoch_losses["ot"]:.4f}',
            'Density': f'{epoch_losses["density"]:.4f}',
            'Energy': f'{epoch_losses["energy"]:.4f}'
        }

        # Add current learning rate if scheduler is used
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            postfix_dict['LR'] = f'{current_lr:.2e}'

        epoch_pbar.set_postfix(postfix_dict)

    return history

