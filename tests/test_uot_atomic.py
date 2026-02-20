import torch
import ot

def test_uot():
    # Source: 2 points. Top point (good), Bottom point (bad)
    # Target: 1 point at the Top.
    # We want source_mass[0] -> 1.0, source_mass[1] -> 0.0
    
    source_mass = torch.tensor([1.0, 1.0], requires_grad=True)
    target_mass = torch.tensor([1.0])
    
    # Distance matrix M: Top->Target is 0.1, Bottom->Target is 10.0
    M = torch.tensor([[0.1], [10.0]])
    
    optimizer = torch.optim.Adam([source_mass], lr=0.1)
    
    for i in range(100):
        optimizer.zero_grad()
        
        # Softplus to ensure positivity
        sm = torch.nn.functional.softplus(source_mass)
        
        # Scale to preserve absolute meaning (mean=1)
        mu = sm / sm.size(0)
        nu = target_mass / target_mass.size(0)
        
        # Clamp to avoid NaN in Sinkhorn log
        mu = torch.clamp(mu, min=1e-5)
        nu = torch.clamp(nu, min=1e-5)
        
        # Normalize M to prevent Sinkhorn overflow
        M_norm = M / M.max()
        
        # Unbalanced Sinkhorn
        # reg is entropy (higher = smoother, more stable)
        # reg_m is marginal penalty (lower = easier to destroy mass)
        loss_ot = ot.unbalanced.sinkhorn_unbalanced2(mu, nu, M_norm, reg=0.1, reg_m=2.0)
        
        # L1 prior to pull everything to 1.0
        # Wait, if L1 is too strong, it ignores OT. 
        loss_l1 = 0.05 * torch.mean(torch.abs(sm - 1.0))
        
        loss = loss_ot + loss_l1
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Iter {i}: sm={sm.detach().numpy()}, loss_ot={loss_ot.item():.4f}, loss_l1={loss_l1.item():.4f}")
            
    print("Final sm:", sm.detach().numpy())

if __name__ == "__main__":
    test_uot()
