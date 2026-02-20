import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import make_dying_example_unif
from mioflow import TimeSeriesDataset, ODEFunc, GrowthRateModel, train_mioflow, infer, compute_uot_growth_rates, pretrain_growth_model

def get_final_losses(history_dict):
    final_losses = {}
    for k, v in history_dict.items():
        if k != 'learning_rates':
            try:
                if hasattr(v, '__len__') and len(v) > 0:
                    final_losses[k] = float(v[-1])
            except (IndexError, TypeError, KeyError):
                continue
    return final_losses

def main():
    print("Generating dying example dataset...")
    df = make_dying_example_unif(n_pts_per_bin=50, seed=223)
    
    # Preprocess Data
    time_pts = sorted(df['samples'].unique())
    time_series_data = []
    
    for t in time_pts:
        pts = df[df['samples'] == t][['d1', 'd2']].values
        time_series_data.append((pts, float(t)))
        
    # Scale Data
    all_pts = np.vstack([x for x, t in time_series_data])
    scaler = StandardScaler()
    scaler.fit(all_pts)
    
    scaled_data = []
    for x, t in time_series_data:
        scaled_data.append((scaler.transform(x), t))
        
    dataset = TimeSeriesDataset(scaled_data)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ode_model = ODEFunc(input_dim=2, hidden_dim=64).to(device)
    growth_model = GrowthRateModel(input_dim=2, hidden_dim=32, use_time=True).to(device)
    
    print("======== Version B: UOT Initialization ========")
    print("1. Computing Unbalanced OT mass mathematically...")
    # Aggressively small tau (0.1) massively rewards dropping mass
    # rather than transporting expensive dead cells across the field
    try:
        uot_masses = compute_uot_growth_rates(
            dataset=dataset, 
            reg_m=[2.0, 100.0],
            div='kl'
        )
    except Exception as e:
        print(f"Error computing UOT growth rates: {e}")
        # Depending on desired error handling, you might want to re-raise,
        # exit, or set uot_masses to a default/empty value.
        # For now, we'll re-raise to halt execution on error.
        raise
    
    print("2. Pre-training the GrowthRateModel to predict UOT masses...")
    pretrain_growth_model(growth_model, dataset, uot_masses, num_epochs=150, learning_rate=1e-3, device=device)
    
    print("3. Training MIOFlow with pre-trained Growth Rate concurrently...")
    history = train_mioflow(
        model=ode_model,
        dataset=dataset,
        num_epochs=100,
        batch_size=None,
        learning_rate=1e-3,
        device=device,
        lambda_ot=1.0,
        lambda_density=0.01,
        lambda_energy=0.2,  # Reduced from 2.0 to allow trajectories to complete
        energy_time_steps=10,
        growth_rate_model=growth_model,
        growth_rate_lr=0.0  # Explicitly freeze the pre-trained growth model
    )
    
    final_losses = get_final_losses(history)
    print("Final losses:", final_losses)
    
    # Plot losses
    epochs = history['epoch']
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['total_loss'], 'k-', linewidth=3, label='Total Loss')
    plt.plot(epochs, history['ot_loss'], 'b-', linewidth=2, label='OT Loss')
    plt.plot(epochs, history['energy_loss'], 'g-', linewidth=2, label='Energy Loss')
    plt.title('MIOFlow with Growth Rate (UOT Init) - Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('tests/growth_rate_uot_losses.png')
    print("Saved loss plot to tests/growth_rate_uot_losses.png")
    
    # Visualize Trajectories
    print("Visualizing Trajectories...")
    X_0_full = dataset.get_initial_condition()
    n_pts = min(100, X_0_full.shape[0])
    indices = torch.randperm(X_0_full.shape[0])[:n_pts]
    X_0_sample = torch.tensor(X_0_full[indices], dtype=torch.float32, device=device)
    
    times = sorted(list(set([t for x, t in scaled_data])))
    t_min, t_max = min(times), max(times)
    t_bins = torch.linspace(t_min, t_max, 100, device=device)
    
    # Calculate predicted mass at start
    with torch.no_grad():
        t_0_sample = torch.zeros(X_0_sample.shape[0], 1, dtype=torch.float32, device=device)
        initial_mass = growth_model(X_0_sample, t_0_sample).cpu().numpy()
        trajectories = infer(x0=X_0_sample, model=ode_model, t_seq=t_bins) 
        trajectories = trajectories.permute(1, 0, 2)
        
    all_points = np.vstack([x for x, t in scaled_data])
    all_times = np.concatenate([[t] * len(x) for x, t in scaled_data])
    
    # Calculate predicted mass for ALL points for the separate plot
    with torch.no_grad():
        all_pts_tensor = torch.tensor(all_points, dtype=torch.float32, device=device)
        all_t_tensor = torch.tensor(all_times, dtype=torch.float32, device=device).view(-1, 1)
        all_predicted_mass = growth_model(all_pts_tensor, all_t_tensor).cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(all_points[:, 0], all_points[:, 1], 
                         c=all_times, cmap='viridis', alpha=0.2, s=20,
                         label='Original data')
    
    # Pre-compute min/max mass over the WHOLE trajectory for consistent scaling
    with torch.no_grad():
        all_masses_for_scaling = []
        for i in range(len(t_bins)):
            t_curr = torch.full((trajectories.size(0), 1), t_bins[i].item(), dtype=torch.float32, device=device)
            mass_curr = growth_model(trajectories[:, i, :], t_curr).cpu().numpy()
            all_masses_for_scaling.append(mass_curr)
            
        all_masses_for_scaling = np.concatenate(all_masses_for_scaling)
        mass_min, mass_max = np.min(all_masses_for_scaling), np.max(all_masses_for_scaling)
        mass_range = mass_max - mass_min if mass_max > mass_min else 1e-6
    
    for i in range(trajectories.size(0)):
        traj = trajectories[i].cpu().numpy()
        
        # Draw trajectory segment by segment to dynamically change linewidth
        for j in range(len(t_bins) - 1):
            with torch.no_grad():
                t_curr = torch.tensor([[t_bins[j].item()]], dtype=torch.float32, device=device)
                pt_curr = torch.tensor([traj[j]], dtype=torch.float32, device=device)
                mass = growth_model(pt_curr, t_curr).item()
                
            # Min-max normalization for extreme visual contrast
            normalized_rate = (mass - mass_min) / mass_range
            # Map 0 -> 1 to linewidth 0.5 -> 6.0
            linewidth = 0.5 + normalized_rate * 5.5
            # Map 0 -> 1 to alpha 0.05 -> 1.0
            alpha = 0.05 + normalized_rate * 0.95
            alpha = min(1.0, max(0.0, float(alpha)))
            
            # EARLY STOPPING: Biology dictates dead cells don't flow.
            # If the predicted mass drops below a small threshold relative to the max, 
            # we terminate the trajectory rendering.
            if normalized_rate < 0.1:
                break
            
            plt.plot(traj[j:j+2, 0], traj[j:j+2, 1], 'k-', alpha=alpha, linewidth=linewidth)
            
        # Draw final arrow using the last segment's style (only if it didn't die immediately)
        if traj.shape[0] >= 2 and normalized_rate >= 0.1:
            dx, dy = traj[j+1, 0] - traj[j, 0], traj[j+1, 1] - traj[j, 1]
            arrow_length = np.sqrt(dx**2 + dy**2)
            if arrow_length > 0:
                scale = 0.1 / arrow_length
                plt.arrow(traj[j, 0], traj[j, 1], dx*scale, dy*scale,
                          head_width=0.05, head_length=0.08, fc='k', ec='k', alpha=alpha)
                          
    plt.colorbar(scatter, label='Time')
    plt.title('MIOFlow Trajectories (Thickness ~ Predicted Mass - UOT Init)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('tests/growth_rate_uot_trajectories.png')
    print("Saved trajectory plot to tests/growth_rate_uot_trajectories.png")
    
    # Predicted mass on ALL points
    plt.figure(figsize=(10, 8))
    mass_scatter = plt.scatter(all_points[:, 0], all_points[:, 1], 
                               c=all_predicted_mass, cmap='viridis', s=20, alpha=0.8)
    plt.colorbar(mass_scatter, label='Predicted Growth Rate')
    plt.title('Predicted Growth Rate Field (UOT Init)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('tests/growth_rate_uot_all_mass.png')
    print("Saved all mass plot to tests/growth_rate_uot_all_mass.png")

if __name__ == "__main__":
    main()
