import torch
import ot

def test_plan_dists():
    source_mass = torch.tensor([1.0, 1.0], requires_grad=True) # [Top (good), Bottom (bad)]
    target_mass = torch.tensor([1.0])
    
    # M: Top->Target is cheap (0.1), Bottom->Target is expensive (10.0)
    M = torch.tensor([[0.1], [10.0]])
    
    optimizer = torch.optim.Adam([source_mass], lr=0.1)
    
    for i in range(100):
        optimizer.zero_grad()
        sm = torch.nn.functional.softplus(source_mass)
        
        # Exact EMD normalizes the probabilities
        mu = sm / sm.sum()
        nu = target_mass / target_mass.sum()
        
        # Get the transport plan completely detached from gradients
        with torch.no_grad():
            plan = ot.emd(mu.detach(), nu.detach(), M) # shape [N, M]
            
            # For each source point i, its average travel distance is:
            # D_i = sum_j(plan[i,j] * M[i,j]) / mu[i]
            # Add eps to prevent div by zero
            C_i = torch.sum(plan * M, dim=1) # cost per source point
            D_i = C_i / (mu.detach() + 1e-8)
            
        # The total loss is just the sum of ( mass_i * Average Distance_i )
        # This is a perfect mathematically uncoupled linear penalty!
        loss_ot = torch.sum(sm * D_i) / sm.size(0)
        
        loss_l1 = 0.5 * torch.mean(torch.abs(sm - 1.0))
        
        loss = loss_ot + loss_l1
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Iter {i}: sm={sm.detach().numpy()}, loss_ot={loss_ot.item():.4f}, loss_l1={loss_l1.item():.4f}")
            
    print("Final sm:", sm.detach().numpy())

if __name__ == "__main__":
    test_plan_dists()
