import torch
import ot

def test_bounded_lse():
    source_mass = torch.tensor([1.0, 1.0], requires_grad=True) # [Top (good), Bottom (bad)]
    target_mass = torch.tensor([1.0])
    
    # M: Top->Target is cheap (0.1), Bottom->Target is expensive (10.0)
    # The ODE Base Error distance is typically 0.5
    # So a good point might have M=0.5. A bad point has M=10.0.
    M_clean = torch.tensor([[0.0], [10.0]])
    M_noisy = torch.tensor([[0.5], [10.0]])
    
    for name, M in [("Clean", M_clean), ("Noisy", M_noisy)]:
        print(f"\n--- Testing {name} M ---")
        sm_var = source_mass.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([sm_var], lr=0.1)
        
        for i in range(100):
            optimizer.zero_grad()
            sm = torch.nn.functional.softplus(sm_var)
            
            epsilon = 0.1
            soft_min_dists = -epsilon * torch.logsumexp(-M / epsilon, dim=1)
            
            # The Magic Fix: Bounded UOT Distance!
            # If distance exceeds threshold (e.g., 2.0), we "destroy" the cell.
            # The cost to destroy is bounded at `threshold`.
            threshold = 2.0
            bounded_dists = torch.clamp(soft_min_dists, max=threshold)
            
            loss_ot = torch.sum(sm * bounded_dists) / sm.size(0)
            
            # Weak L1 prior ensures good points stay at 1.0, recovering from ODE base error
            # If base error M=0.5, OT gradient on sm is 0.5.
            # If L1 is STRONGER than base error (e.g. 1.0), it pulls sm back to 1.0!
            loss_l1 = 1.0 * torch.mean(torch.abs(sm - 1.0))
            
            loss = loss_ot + loss_l1
            loss.backward()
            optimizer.step()
            
            if i % 20 == 0:
                print(f"Iter {i}: sm={sm.detach().numpy()}, cost={loss_ot.item():.4f}, loss_l1={loss_l1.item():.4f}")
                
        print("Final sm:", sm.detach().numpy())

if __name__ == "__main__":
    test_bounded_lse()
