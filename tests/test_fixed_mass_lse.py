import torch
import ot

def test_fixed_mass_lse():
    source_mass = torch.tensor([1.0, 1.0], requires_grad=True) # [Top (good), Bottom (bad)]
    target_mass = torch.tensor([1.0])
    
    # M: Top->Target is cheap (0.1), Bottom->Target is expensive (10.0)
    M = torch.tensor([[0.1], [10.0]])
    
    optimizer = torch.optim.Adam([source_mass], lr=0.1)
    
    for i in range(100):
        optimizer.zero_grad()
        sm = torch.nn.functional.softplus(source_mass)
        tm = target_mass
        
        # Soft-min distance from each source point to ALL target points
        # If a point is far from EVERYTHING, its min_dist is large.
        # This is a localized proxy for OT cost that doesn't couple the points via normalization!
        epsilon = 0.1
        
        # LogSumExp to find the soft-minimum distance to any target point
        # shape [source_len, target_len] -> min over target_len (axis 1)
        soft_min_dists = -epsilon * torch.logsumexp(-M / epsilon, dim=1)
        
        # Total cost is the mass of each point * its individual distance to the closest target
        true_cost = torch.sum(sm * soft_min_dists)
        
        # Add L1 penalty to anchor the good points at 1.0
        loss_l1 = 0.5 * torch.mean(torch.abs(sm - 1.0))
        
        loss = true_cost + loss_l1
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Iter {i}: sm={sm.detach().numpy()}, cost={true_cost.item():.4f}, loss_l1={loss_l1.item():.4f}")
            
    print("Final sm:", sm.detach().numpy())

if __name__ == "__main__":
    test_fixed_mass_lse()
