import torch
import ot

def test_emd():
    source_mass = torch.tensor([1.0, 1.0], requires_grad=True)
    target_mass = torch.tensor([1.0])
    M = torch.tensor([[0.1], [10.0]])
    
    optimizer = torch.optim.Adam([source_mass], lr=0.1)
    
    for i in range(100):
        optimizer.zero_grad()
        sm = torch.nn.functional.softplus(source_mass)
        
        # Exact EMD requires normalized mu and nu
        mu = sm / sm.sum()
        nu = target_mass / target_mass.sum()
        
        loss_ot = ot.emd2(mu, nu, M)
        # Scale back up
        loss_ot = loss_ot * (sm.sum() / sm.size(0))
        
        loss_l1 = 0.5 * torch.mean(torch.abs(sm - 1.0))
        
        loss = loss_ot + loss_l1
        loss.backward()
        optimizer.step()
        
    print("Final sm (EMD scaled):", sm.detach().numpy())

if __name__ == "__main__":
    test_emd()
