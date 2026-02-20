import torch
import ot

def test_fixed_mass():
    source_mass = torch.tensor([1.0, 1.0], requires_grad=True) # [Top (good), Bottom (bad)]
    target_mass = torch.tensor([1.0])
    
    # M: Top->Target is cheap (0.1), Bottom->Target is expensive (10.0)
    M = torch.tensor([[0.1], [10.0]])
    
    optimizer = torch.optim.Adam([source_mass], lr=0.1)
    
    for i in range(100):
        optimizer.zero_grad()
        sm = torch.nn.functional.softplus(source_mass)
        
        # Exact EMD requires normalized mu and nu
        mu = sm / sm.sum()
        nu = target_mass / target_mass.sum()
        
        # We need the transport PLAN to correctly attribute cost to the source points!
        plan = ot.emd(mu.detach(), nu.detach(), M) # [source_len, target_len]
        
        # Attribute the actual cost back to the unnormalized source mass
        # Multiply the transport plan by the true unnormalized mass
        true_cost = torch.sum((plan * M) * sm.unsqueeze(1))
        
        loss_l1 = 0.05 * torch.mean(torch.abs(sm - 1.0))
        
        loss = true_cost + loss_l1
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Iter {i}: sm={sm.detach().numpy()}, cost={true_cost.item():.4f}, loss_l1={loss_l1.item():.4f}")
            
    print("Final sm:", sm.detach().numpy())

if __name__ == "__main__":
    test_fixed_mass()
