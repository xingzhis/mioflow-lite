import torch
import ot

source_mass = torch.tensor([1.0, 1.0], requires_grad=True) # Top, Bottom
target_mass = torch.tensor([1.0, 1.0])

mu = source_mass
nu = target_mass

M = torch.tensor([[0.1, 0.1], [10.0, 10.0]])

# Unbalanced Sinkhorn
loss_ot = ot.unbalanced.sinkhorn_unbalanced2(mu, nu, M, reg=1.0, reg_m=1.0, method='sinkhorn')
loss_l1 = 0.1 * torch.mean(torch.abs(source_mass - 1.0))
loss = loss_ot + loss_l1

loss.backward()

print("OT Loss:", loss_ot.item())
print("Gradient of source_mass:", source_mass.grad)

optimizer = torch.optim.Adam([source_mass], lr=0.1)
for i in range(100):
    optimizer.zero_grad()
    mu = torch.nn.functional.softplus(source_mass) + 1e-4
    loss_ot = ot.unbalanced.sinkhorn_unbalanced2(mu, nu, M, reg=1.0, reg_m=1.0, method='sinkhorn')
    loss_l1 = 0.1 * torch.mean(torch.abs(torch.nn.functional.softplus(source_mass) + 1e-4 - 1.0))
    loss = loss_ot + loss_l1
    loss.backward()
    optimizer.step()

final_mass = torch.nn.functional.softplus(source_mass) + 1e-4
print("Final mass:", final_mass.detach())
